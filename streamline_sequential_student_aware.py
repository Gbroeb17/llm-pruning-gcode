"""Heal every maximal removed run using its preceding retained decoder layer.

Read the existing pruneme_log.json; do not prune or use QLoRA. Each replacement
learns current student input(start-1) -> original teacher output(end). Train one
replacement at a time, then save all healed layers in the pruned checkpoint.
"""

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path


def build_healing_plan(pruning_log):
    """Use original zero-based indices and coalesce adjacent deleted blocks."""
    num_original = pruning_log.get("num_layers_original")
    removed = pruning_log.get("layers_removed")
    if type(num_original) is not int or num_original < 1:
        raise ValueError("Pruning log needs a positive integer num_layers_original.")
    if not isinstance(removed, list) or any(type(i) is not int for i in removed):
        raise ValueError("Pruning log needs an integer list layers_removed.")
    if len(set(removed)) != len(removed) or any(i < 0 or i >= num_original for i in removed):
        raise ValueError("Removed indices must be unique and within the original model.")
    removed = sorted(removed)
    removed_set = set(removed)
    kept = [i for i in range(num_original) if i not in removed_set]
    if not kept:
        raise ValueError("No retained decoder layers remain.")
    if pruning_log.get("layers_kept", kept) != kept:
        raise ValueError("layers_kept does not match the complement of layers_removed.")
    if pruning_log.get("new_to_original_layer", kept) != kept:
        raise ValueError("new_to_original_layer disagrees with layers_removed.")
    if pruning_log.get("total_prune_layers", len(removed)) != len(removed):
        raise ValueError("total_prune_layers disagrees with layers_removed.")
    if removed and removed[0] == 0:
        raise ValueError("Original layer 0 was removed: it has no preceding layer to heal.")

    runs = []
    for index in removed:
        if runs and index == runs[-1][1] + 1:
            runs[-1][1] = index
        else:
            runs.append([index, index])
    original_to_pruned = {original: current for current, original in enumerate(kept)}
    plan = [
        {
            "segment": segment,
            "removed_start": start,
            "removed_end": end,
            "removed_count": end - start + 1,
            "replacement_original_idx": start - 1,
            "replacement_pruned_idx": original_to_pruned[start - 1],
            "teacher_input_layer": start - 1,
            "teacher_output_layer": end,
        }
        for segment, (start, end) in enumerate(runs, start=1)
    ]
    return kept, plan


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original_model_dir", default="/media/ssd2/llm-pruning/shared/models/granite_gcode_merged_best_new")
    parser.add_argument("--pruned_model_dir", type=Path,default=Path("/media/ssd2/llm-pruning/cs_115522107/llm-pruning-gcode/experiments/iterative_pruning/outputs/granite3b-pruneme-iterative-multiblock-b1-n16-skip16"),)
    parser.add_argument("--pruning_log", type=Path, help="Default: pruned_model_dir/pruneme_log.json")
    parser.add_argument("--train_jsonl", default="/media/ssd2/llm-pruning/shared/datasets/training_data/dataset_no_rule/train.jsonl")
    parser.add_argument("--output_dir", type=Path, help="Default: sibling of pruned model with -streamline suffix")
    parser.add_argument("--plan_only", action="store_true", help="Print detected segments without loading models or training.")
    parser.add_argument("--max_seq_len", type=int, default=8192)
    parser.add_argument("--limit_samples", type=int, default=None)
    parser.add_argument("--min_seq_len", type=int, default=1024)
    parser.add_argument("--short_threshold", type=int, default=3600)
    parser.add_argument("--long_threshold", type=int, default=20000)
    parser.add_argument("--short_ratio", type=float, default=0.5)
    parser.add_argument("--medium_ratio", type=float, default=0.3)
    parser.add_argument("--long_ratio", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_ratio", type=float, default=0.01)
    parser.add_argument("--cosine_training_ratio", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device", default=None, help="Default: cuda if available, otherwise cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_args(args):
    for name in ["max_seq_len", "min_seq_len", "batch_size", "epochs", "grad_accum"]:
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive.")
    if args.limit_samples is not None and args.limit_samples <= 0:
        raise ValueError("limit_samples must be positive or omitted.")
    if not 0 < args.min_seq_len <= args.short_threshold < args.long_threshold:
        raise ValueError("Expected min_seq_len <= short_threshold < long_threshold.")
    ratios = [args.short_ratio, args.medium_ratio, args.long_ratio]
    if any(not math.isfinite(r) or r < 0 for r in ratios) or not math.isclose(sum(ratios), 1.0):
        raise ValueError("Sampling ratios must be non-negative and sum to 1.")
    if not 0 <= args.min_lr <= args.lr or args.lr <= 0:
        raise ValueError("Expected 0 <= min_lr <= lr and lr > 0.")
    if not 0 <= args.warmup_ratio < 1 or not 0 < args.cosine_training_ratio <= 1:
        raise ValueError("Expected 0 <= warmup_ratio < 1 and 0 < cosine_training_ratio <= 1.")


class JsonlGCodeDataset:
    """Keep the supplied healing script's sampling and random-chunk behavior."""

    def __init__(self, tokenizer, args):
        self.max_seq_len = args.max_seq_len
        self.rng = random.Random(args.seed)
        raw = []
        with open(args.train_jsonl, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                text = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
                ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                raw.append({"input_ids": ids, "tok_len": len(ids)})
        pool = [row for row in raw if row["tok_len"] >= args.min_seq_len]
        short = [row for row in pool if row["tok_len"] < args.short_threshold]
        medium = [row for row in pool if args.short_threshold <= row["tok_len"] < args.long_threshold]
        long = [row for row in pool if row["tok_len"] >= args.long_threshold]
        if args.limit_samples:
            chosen = []
            for bucket, ratio in [(short, args.short_ratio), (medium, args.medium_ratio), (long, args.long_ratio)]:
                chosen.extend(self.rng.sample(bucket, min(int(args.limit_samples * ratio), len(bucket))))
            while len(chosen) < args.limit_samples:
                chosen_ids = {id(row) for row in chosen}
                leftover = [row for row in pool if id(row) not in chosen_ids]
                if not leftover:
                    break
                chosen.append(self.rng.choice(leftover))
        else:
            chosen = list(pool)
        self.rng.shuffle(chosen)
        self.samples = [row["input_ids"] for row in chosen[:args.limit_samples]]
        if not self.samples:
            raise ValueError("No eligible healing samples remain.")
        self.chunk_rng_state = self.rng.getstate()
        print(f"Pool: {len(pool)}, buckets: {len(short)}/{len(medium)}/{len(long)}; healing samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        import torch

        ids = self.samples[index]
        if len(ids) > self.max_seq_len:
            start = self.rng.randint(0, len(ids) - self.max_seq_len)
            ids = ids[start:start + self.max_seq_len]
        return torch.tensor(ids, dtype=torch.long)


class PadCollator:
    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, samples):
        import torch

        max_len = max(ids.numel() for ids in samples)
        input_ids = torch.full((len(samples), max_len), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row, ids in enumerate(samples):
            input_ids[row, :ids.numel()] = ids
            attention_mask[row, :ids.numel()] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def get_layers(model):
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("Expected a Granite/LLaMA-style model.model.layers.")
    return model.model.layers


def detach_tree(value):
    """Keep native forward arguments, including masks and rotary embeddings."""
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, tuple):
        return tuple(detach_tree(item) for item in value)
    if isinstance(value, list):
        return [detach_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: detach_tree(item) for key, item in value.items()}
    return value


class _CaptureComplete(Exception):
    """Private control flow: stop the real backbone forward at a hook."""


def normalize_layer_indices(model):
    """Use contiguous student indices for caches and per-layer mask selection."""
    layers = get_layers(model)
    model.config.num_hidden_layers = len(layers)
    model.model.config.num_hidden_layers = len(layers)
    for index, layer in enumerate(layers):
        for module in layer.modules():
            if hasattr(module, "layer_idx"):
                module.layer_idx = index


def teacher_target(teacher, batch, segment):
    import torch

    cache = {}

    def save_output(module, positional, output):
        cache["target"] = detach_tree(output[0] if isinstance(output, tuple) else output)
        raise _CaptureComplete()

    handle = get_layers(teacher)[segment["teacher_output_layer"]].register_forward_hook(save_output)
    try:
        teacher.eval()
        with torch.no_grad():
            try:
                teacher.model(**batch, use_cache=False, output_hidden_states=False, return_dict=True)
            except _CaptureComplete:
                pass
        if "target" not in cache:
            raise RuntimeError("Teacher output hook did not run.")
        return cache["target"]
    finally:
        handle.remove()


def student_layer_input(student, batch, segment, device):
    """Run current student prefix, streaming one CPU decoder layer at a time.

    Capture the actual args/kwargs; never synthesize RoPE, positions, or masks.
    Only standard dense Granite/LLaMA-style backbones are supported (no device_map,
    quantized weights, or Accelerate offload hooks). Student remains frozen/eval.
    """
    import torch

    cache, handles, resident = {}, [], []
    backbone = student.model
    layers = get_layers(student)
    index = segment["replacement_pruned_idx"]
    auxiliary = [child for child in backbone.children() if child is not layers]
    original_buffers = dict(backbone._buffers)

    def load_layer(module, positional):
        resident.append(module)
        module.to(device)

    def unload_layer(module, positional, output):
        module.cpu()
        resident.remove(module)

    def save_input(module, positional, kwargs):
        cache["args"] = detach_tree(positional)
        cache["kwargs"] = detach_tree(kwargs)
        for name in ("past_key_value", "past_key_values"):
            if cache["kwargs"].get(name) is not None:
                raise RuntimeError("Unexpected attention cache with use_cache=False.")
        raise _CaptureComplete()

    try:
        student.eval()
        for child in auxiliary:
            child.to(device)
        for name, value in original_buffers.items():
            if value is not None:
                backbone._buffers[name] = value.to(device)
        for layer in layers[:index]:
            handles.append(layer.register_forward_pre_hook(load_layer))
            handles.append(layer.register_forward_hook(unload_layer))
        handles.append(layers[index].register_forward_pre_hook(save_input, with_kwargs=True))
        with torch.no_grad():
            try:
                backbone(**batch, use_cache=False, output_hidden_states=False, return_dict=True)
            except _CaptureComplete:
                pass
        if set(cache) != {"args", "kwargs"}:
            raise RuntimeError("Student input hook did not run.")
        return cache["args"], cache["kwargs"]
    finally:
        for handle in handles:
            handle.remove()
        for layer in resident:
            layer.cpu()
        for child in auxiliary:
            child.cpu()
        for name, value in original_buffers.items():
            backbone._buffers[name] = value


def run_layer(layer, positional, kwargs):
    output = layer(*positional, **kwargs)
    return output[0] if isinstance(output, tuple) else output


def masked_mse(prediction, target, attention_mask):
    import torch.nn.functional as functional

    if prediction.shape != target.shape or prediction.shape[:2] != attention_mask.shape:
        raise ValueError("Prediction, teacher target, and token mask shapes disagree.")
    valid = attention_mask.bool()
    if not valid.any():
        raise ValueError("Cannot compute MSE on an entirely padded batch.")
    return functional.mse_loss(prediction[valid].float(), target[valid].float())


def make_lr_lambda(total_steps, args):
    scheduler_steps = max(1, int(total_steps * args.cosine_training_ratio))
    warmup_steps = min(scheduler_steps, int(scheduler_steps * args.warmup_ratio))

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step + 1) / warmup_steps
        progress = min(1.0, (current_step - warmup_steps) / max(1, scheduler_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        minimum = args.min_lr / args.lr
        return minimum + (1.0 - minimum) * cosine

    return lr_lambda


def train_streamline_replacement(teacher, student, source_layer, dataloader, segment, args, device):
    import torch
    from tqdm import tqdm

    replacement = copy.deepcopy(source_layer).to(device)
    source_dtype = next(replacement.parameters()).dtype
    if args.dtype == "fp16":
        replacement = replacement.float()
    replacement.requires_grad_(True)
    replacement.train()
    optimizer = torch.optim.AdamW(replacement.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = math.ceil(len(dataloader) / args.grad_accum) * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, make_lr_lambda(total_steps, args))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.dtype == "fp16"))
    best_loss = float("inf")
    best_state = None
    history = []
    optimizer_steps = 0
    for epoch in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_batches = 0
        for step, batch in enumerate(tqdm(dataloader, desc=f"Segment {segment['segment']} epoch {epoch + 1}")):
            batch = {key: value.to(device) for key, value in batch.items()}
            target = teacher_target(teacher, batch, segment)
            positional, kwargs = student_layer_input(student, batch, segment, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=(device.type == "cuda" and args.dtype == "fp16")):
                prediction = run_layer(replacement, positional, kwargs)
                raw_loss = masked_mse(prediction, target, batch["attention_mask"])
            if not torch.isfinite(raw_loss):
                raise ValueError("Non-finite MSE; inspect dtype, learning rate, and activations.")
            # Flush a partial final accumulation window rather than discarding it.
            window_start = (step // args.grad_accum) * args.grad_accum
            window_size = min(args.grad_accum, len(dataloader) - window_start)
            scaler.scale(raw_loss / window_size).backward()
            if (step + 1) % args.grad_accum == 0 or step + 1 == len(dataloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(replacement.parameters(), 1.0, error_if_nonfinite=True)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous_scale:
                    scheduler.step()
                    optimizer_steps += 1
                optimizer.zero_grad(set_to_none=True)
            total_loss += raw_loss.item()
            total_batches += 1
            del positional, kwargs, target, prediction, raw_loss
        average = total_loss / total_batches
        history.append({"epoch": epoch + 1, "train_mse": average, "lr": scheduler.get_last_lr()[0],
                        "optimizer_steps": optimizer_steps})
        print(f"Segment {segment['segment']} epoch {epoch + 1}: train_mse={average:.6f}")
        if average < best_loss:
            best_loss = average
            best_state = {key: value.detach().cpu().clone() for key, value in replacement.state_dict().items()}
    replacement.load_state_dict(best_state)
    replacement = replacement.to(device="cpu", dtype=source_dtype).eval()
    replacement.requires_grad_(False)
    return replacement, history


def save_healed_model(pruned, tokenizer, output_dir, kept):
    layers = get_layers(pruned)
    if len(layers) != len(kept):
        raise ValueError("Healed model layer count changed unexpectedly.")
    pruned.config.num_hidden_layers = len(layers)
    pruned.model.config.num_hidden_layers = len(layers)
    pruned.config.output_hidden_states = False
    for index, layer in enumerate(layers):
        if hasattr(layer, "layer_idx"):
            layer.layer_idx = index
        attention = getattr(layer, "self_attn", None)
        if attention is not None and hasattr(attention, "layer_idx"):
            attention.layer_idx = index
    pruned.eval()
    pruned.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)


def main():
    args = parse_args()
    validate_args(args)
    log_path = args.pruning_log or args.pruned_model_dir / "pruneme_log.json"
    with log_path.open(encoding="utf-8") as handle:
        pruning_log = json.load(handle)
    kept, plan = build_healing_plan(pruning_log)
    print(f"Detected {len(plan)} continuous removed segments; heal {len(plan)} layers")
    for segment in plan:
        print(f"  Removed original L{segment['removed_start']}..L{segment['removed_end']}; "
              f"train original L{segment['replacement_original_idx']} "
              f"= pruned index {segment['replacement_pruned_idx']}")
    if args.plan_only:
        return
    output_dir = args.output_dir or args.pruned_model_dir.with_name(args.pruned_model_dir.name + "-streamline-student-aware")
    if output_dir.resolve() in {args.pruned_model_dir.resolve(), Path(args.original_model_dir).resolve()}:
        raise ValueError("output_dir must not overwrite the original or pruned input model.")

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.time()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    if args.dtype == "fp16" and device.type != "cuda":
        raise ValueError("fp16 healing requires CUDA; use fp32 for CPU runs.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    tokenizer = AutoTokenizer.from_pretrained(args.original_model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer needs a pad or EOS token.")
    tokenizer.padding_side = "right"
    dataset = JsonlGCodeDataset(tokenizer, args)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.original_model_dir, torch_dtype=dtype, device_map=None,
        trust_remote_code=True, attn_implementation="sdpa",
    ).to(device).eval()
    teacher.requires_grad_(False)
    teacher.config.use_cache = False
    # Teacher and replacement stay on device; student prefix layers stream from CPU.
    pruned = AutoModelForCausalLM.from_pretrained(
        args.pruned_model_dir, torch_dtype=dtype, device_map=None,
        trust_remote_code=True, attn_implementation="sdpa",
    ).cpu().eval()
    pruned.requires_grad_(False)
    pruned.config.use_cache = False
    normalize_layer_indices(pruned)
    if len(get_layers(teacher)) != pruning_log["num_layers_original"] or len(get_layers(pruned)) != len(kept):
        raise ValueError("Model layer counts do not match the pruning log.")
    if teacher.config.hidden_size != pruned.config.hidden_size:
        raise ValueError("Original and pruned model hidden dimensions differ.")
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "method": "sequential student-aware Streamline MSE, one predecessor per maximal removed run",
        "index_base": 0,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "pruning_log": str(log_path),
        "layers_kept": kept,
        "num_healed_layers": len(plan),
        "training_samples": len(dataset),
        "segments": [],
    }
    for segment in plan:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        dataset.rng.setstate(dataset.chunk_rng_state)
        generator = torch.Generator().manual_seed(args.seed)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                collate_fn=PadCollator(tokenizer), generator=generator, num_workers=0)
        healed, history = train_streamline_replacement(
            teacher, pruned, get_layers(pruned)[segment["replacement_pruned_idx"]], dataloader, segment, args, device,
        )
        get_layers(pruned)[segment["replacement_pruned_idx"]] = healed
        checkpoint_name = f"healed_layer_original_{segment['replacement_original_idx']}.pt"
        torch.save(healed.state_dict(), output_dir / checkpoint_name)
        report["segments"].append(dict(segment, epoch_history=history, checkpoint=checkpoint_name))
        with (output_dir / "streamline_multiblock_log.json").open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    save_healed_model(pruned, tokenizer, output_dir, kept)
    with (output_dir / "pruneme_log.json").open("w", encoding="utf-8") as handle:
        json.dump(pruning_log, handle, indent=2)
    report["elapsed_seconds"] = time.time() - started
    report["peak_cuda_memory_gib"] = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None
    report["complete"] = True
    with (output_dir / "streamline_multiblock_log.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(f"Saved healed model and logs: {output_dir}")


if __name__ == "__main__":
    main()
