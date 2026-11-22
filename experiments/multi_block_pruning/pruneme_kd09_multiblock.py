"""Multi-block pruning of the existing kd06_seed44 GPTQ 2-bit checkpoint.

Preserve the supplied multi-block baseline's sampling, score indexing, and
greedy selection. Export retained packed tensors from the original files,
not the inference backend's potentially converted in-memory representation.
No dequantization, requantization, training, or weight averaging is performed.
"""

import argparse
import gc
import json
import math
import random
import re
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", type=Path, default=Path("/media/ssd2/llm-pruning/shared/models/kd06_seed44"))
    parser.add_argument("--train_jsonl", type=Path, default=Path("/media/ssd2/llm-pruning/shared/datasets/training_data/dataset_no_rule/train.jsonl"))
    parser.add_argument("--block_size", type=int, default=12)
    parser.add_argument("--prune_layers", type=int, default=12)
    parser.add_argument("--num_calibration_samples", type=int, default=128)
    parser.add_argument("--max_seq_length", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=Path)
    return parser.parse_args()


def validate_source(config, args):
    quantization = config.get("quantization_config", {})
    if config.get("model_type") != "llama":
        raise ValueError("This exporter is scoped to the supplied Llama-format kd06 model.")
    if quantization.get("quant_method") != "gptq" or quantization.get("bits") != 2:
        raise ValueError("Expected the supplied GPTQ 2-bit checkpoint.")
    if quantization.get("dynamic") or quantization.get("modules_in_block_to_quantize"):
        raise ValueError("Per-module quantization rules need explicit index remapping; not supported here.")
    num_layers = config["num_hidden_layers"]
    if not 0 < args.prune_layers < num_layers:
        raise ValueError("prune_layers must be positive and smaller than the original layer count.")
    if args.block_size <= 0 or args.prune_layers % args.block_size:
        raise ValueError("prune_layers must be divisible by a positive block_size.")
    if args.num_calibration_samples <= 0 or args.max_seq_length <= 0:
        raise ValueError("Calibration sample count and maximum sequence length must be positive.")


def build_calibration_data(tokenizer, args):
    raw = []
    with args.train_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            text = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            raw.append({"input_ids": ids, "tok_len": len(ids)})
    pool = [row for row in raw if row["tok_len"] >= 1024]
    short = [row for row in pool if row["tok_len"] < 3600]
    medium = [row for row in pool if 3600 <= row["tok_len"] < 20000]
    long = [row for row in pool if row["tok_len"] >= 20000]
    rng = random.Random(args.seed)
    chosen = []
    for bucket, ratio in [(short, 0.5), (medium, 0.3), (long, 0.2)]:
        chosen.extend(rng.sample(bucket, min(int(args.num_calibration_samples * ratio), len(bucket))))
    while len(chosen) < args.num_calibration_samples:
        leftover = [row for row in pool if row not in chosen]
        if not leftover:
            break
        chosen.append(rng.choice(leftover))
    rng.shuffle(chosen)
    chosen = chosen[:args.num_calibration_samples]
    if not chosen:
        raise ValueError("No eligible calibration samples remain.")
    print(f"Pool: {len(pool)}, buckets: {len(short)}/{len(medium)}/{len(long)}")
    print(f"Selected {len(chosen)} calibration samples")
    return chosen


def angular_distance(left, right):
    import torch

    left_norm = left / torch.norm(left, dim=-1, keepdim=True)
    right_norm = right / torch.norm(right, dim=-1, keepdim=True)
    return torch.acos((left_norm * right_norm).sum(-1).clamp(-1, 1)) / torch.pi


def compute_block_scores(model, samples, args):
    import numpy as np
    import torch
    from tqdm import tqdm

    num_layers = len(model.model.layers)
    all_distances = [[] for _ in range(num_layers - args.block_size)]
    device = model.model.embed_tokens.weight.device
    with torch.no_grad():
        for sample in tqdm(samples, desc="Calibration"):
            input_ids = torch.tensor(sample["input_ids"][:args.max_seq_length],
                                     dtype=torch.long, device=device).unsqueeze(0)
            outputs = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                            output_hidden_states=True, use_cache=False)
            # Intentionally retain the supplied baseline's score indexing.
            # score[i] compares output(i) to output(i+N), while selection removes
            # i..i+N-1. Correcting this offset only here would change the baseline.
            last_tokens = [hidden[:, -1, :] for hidden in outputs.hidden_states][1:]
            if len(last_tokens) != num_layers:
                raise ValueError("Unexpected number of decoder hidden states.")
            for index in range(num_layers - args.block_size):
                value = angular_distance(last_tokens[index], last_tokens[index + args.block_size]).mean().item()
                if not math.isfinite(value):
                    raise ValueError(f"Non-finite calibration score at block {index}.")
                all_distances[index].append(value)
            del outputs, last_tokens
    return [float(np.mean(values)) for values in all_distances]


def select_blocks(scores, num_layers, block_size, prune_layers):
    if block_size <= 0 or not 0 < prune_layers < num_layers or prune_layers % block_size:
        raise ValueError("Invalid block size or pruning budget.")
    if len(scores) != num_layers - block_size or any(not math.isfinite(score) for score in scores):
        raise ValueError("Invalid baseline candidate scores.")
    candidates = [{"start": start, "end": start + block_size - 1, "distance": score}
                  for start, score in enumerate(scores)]
    candidates.sort(key=lambda item: (item["distance"], item["start"]))
    selected = []
    removed = set()
    for candidate in candidates:
        indices = set(range(candidate["start"], candidate["end"] + 1))
        if indices & removed:
            continue
        selected.append(candidate)
        removed.update(indices)
        if len(removed) == prune_layers:
            break
    if len(removed) != prune_layers:
        raise ValueError("Greedy selection could not meet the budget with non-overlapping blocks.")
    selected.sort(key=lambda item: item["start"])
    kept = [index for index in range(num_layers) if index not in removed]
    return selected, sorted(removed), kept


def remap_checkpoint_key(key, original_to_pruned):
    match = re.match(r"^model\.layers\.(\d+)\.(.+)$", key)
    if match is None:
        return key
    original = int(match.group(1))
    if original not in original_to_pruned:
        return None
    return f"model.layers.{original_to_pruned[original]}.{match.group(2)}"


def export_packed_checkpoint(source_dir, output_dir, source_config, kept,
                             safe_open_fn=None, save_file_fn=None):
    """Retain original GPTQ qweight/qzeros/scales/g_idx tensors byte-for-byte."""
    if safe_open_fn is None:
        from safetensors import safe_open as safe_open_fn
    if save_file_fn is None:
        from safetensors.torch import save_file as save_file_fn
    if output_dir.resolve() == source_dir.resolve():
        raise ValueError("The output directory must differ from the source.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Use an empty output directory to avoid mixing checkpoint shards.")
    if not kept or kept != sorted(set(kept)) or any(i < 0 or i >= source_config["num_hidden_layers"] for i in kept):
        raise ValueError("Kept indices must be sorted, unique, and within the source model.")
    with (source_dir / "model.safetensors.index.json").open(encoding="utf-8") as handle:
        source_index = json.load(handle)
    weight_map = source_index["weight_map"]
    original_to_pruned = {original: index for index, original in enumerate(kept)}
    by_shard = {}
    seen_layers = set()
    for key, shard in weight_map.items():
        if Path(shard).name != shard or not shard.endswith(".safetensors"):
            raise ValueError("Expected plain safetensors shard filenames.")
        match = re.match(r"^model\.layers\.(\d+)\.", key)
        if match:
            seen_layers.add(int(match.group(1)))
        mapped = remap_checkpoint_key(key, original_to_pruned)
        if mapped is not None:
            by_shard.setdefault(shard, []).append((key, mapped))
    if seen_layers != set(range(source_config["num_hidden_layers"])):
        raise ValueError("Source index layer names disagree with config.num_hidden_layers.")
    output_dir.mkdir(parents=True, exist_ok=True)
    new_weight_map = {}
    total_size = 0
    shard_names = sorted(by_shard)
    for position, source_name in enumerate(shard_names, start=1):
        output_name = f"model-{position:05d}-of-{len(shard_names):05d}.safetensors"
        with safe_open_fn(str(source_dir / source_name), framework="pt", device="cpu") as reader:
            tensors = {}
            for original_key, mapped_key in by_shard[source_name]:
                tensor = reader.get_tensor(original_key)
                tensors[mapped_key] = tensor
                total_size += tensor.numel() * tensor.element_size()
                new_weight_map[mapped_key] = output_name
            metadata = reader.metadata() or {"format": "pt"}
            save_file_fn(tensors, str(output_dir / output_name), metadata=metadata)
        del tensors
    index = {"metadata": dict(source_index.get("metadata", {}), total_size=total_size),
             "weight_map": new_weight_map}
    with (output_dir / "model.safetensors.index.json").open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2)
    config = dict(source_config, num_hidden_layers=len(kept), output_hidden_states=False)
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    for pattern in ["tokenizer*", "special_tokens_map.json", "added_tokens.json", "vocab.*",
                    "merges.txt", "chat_template*", "generation_config.json", "quantization_config.json"]:
        for path in source_dir.glob(pattern):
            if path.is_file():
                shutil.copy2(path, output_dir / path.name)
    provenance = source_dir / "rerun_metadata.json"
    if provenance.is_file():
        shutil.copy2(provenance, output_dir / "source_rerun_metadata.json")
    return total_size


def load_quantized_model(path):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.float16, device_map={"": 0}, local_files_only=True,
    ).eval()
    print(f"Loaded {len(model.model.layers)} layers; quantized projection: {type(model.model.layers[0].self_attn.q_proj)}")
    return model


def prune_in_memory(model, kept):
    import torch

    model.model.layers = torch.nn.ModuleList([model.model.layers[index] for index in kept])
    model.config.num_hidden_layers = len(kept)
    model.model.config.num_hidden_layers = len(kept)
    model.config.output_hidden_states = False
    for index, layer in enumerate(model.model.layers):
        if hasattr(layer, "layer_idx"):
            layer.layer_idx = index
        if hasattr(layer.self_attn, "layer_idx"):
            layer.self_attn.layer_idx = index


def main():
    args = parse_args()
    with (args.model_dir / "config.json").open(encoding="utf-8") as handle:
        source_config = json.load(handle)
    validate_source(source_config, args)
    output_dir = args.output_dir or Path(__file__).resolve().parent / "outputs" / (
        f"kd06_seed44-gptq2bit-multiblock-b{args.block_size}-n{args.prune_layers // args.block_size}-skip{args.prune_layers}"
    )
    if output_dir.resolve() == args.model_dir.resolve() or (output_dir.exists() and any(output_dir.iterdir())):
        raise ValueError("Choose a new, empty output directory, separate from the source model.")
    import numpy as np
    import torch
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise ValueError("This script expects the tested CUDA GPTQ environment.")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = load_quantized_model(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    num_layers = len(model.model.layers)
    if num_layers != source_config["num_hidden_layers"]:
        raise ValueError("Loaded model layer count disagrees with the source config.")
    print("Scoring uses the original multi-block baseline indexing (including its known one-layer offset).")
    samples = build_calibration_data(tokenizer, args)
    scores = compute_block_scores(model, samples, args)
    blocks, removed, kept = select_blocks(scores, num_layers, args.block_size, args.prune_layers)
    print(f"Selected blocks: {blocks}")
    print(f"Layers removed: {removed}")
    print(f"Layers kept: {kept}")
    prune_in_memory(model, kept)
    smoke_ids = torch.tensor(samples[0]["input_ids"][:16], dtype=torch.long, device="cuda:0").unsqueeze(0)
    smoke_mask = torch.ones_like(smoke_ids)
    with torch.no_grad():
        reference = model(input_ids=smoke_ids, attention_mask=smoke_mask, use_cache=False).logits[:, -1, :].float().cpu()
    if not torch.isfinite(reference).all():
        raise ValueError("Pruned in-memory model produced non-finite logits.")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    packed_bytes = export_packed_checkpoint(args.model_dir, output_dir, source_config, kept)
    log = {
        "model_id": str(args.model_dir), "algorithm": "PruneMe baseline angular scoring + greedy non-overlap selection",
        "pruning_type": "multi_block", "index_base": 0, "num_layers_original": num_layers,
        "block_size": args.block_size, "num_blocks": len(blocks), "total_prune_layers": len(removed),
        "selected_blocks": blocks, "layers_removed": removed, "layers_kept": kept,
        "new_to_original_layer": kept,
        "adjacent_pairs": [{"left_block": [a["start"], a["end"]], "right_block": [b["start"], b["end"]]}
                           for a, b in zip(blocks, blocks[1:]) if a["end"] + 1 == b["start"]],
        "all_block_distances": [{"block_start": i, "block_end": i + args.block_size - 1, "avg_distance": score}
                                for i, score in enumerate(scores)],
        "quantization_config": source_config["quantization_config"],
        "export": "original packed checkpoint tensors, retained and reindexed without requantization",
        "packed_tensor_bytes": packed_bytes,
        "calibration": {"num_samples": len(samples), "max_seq_length": args.max_seq_length,
                        "data_path": str(args.train_jsonl), "seed": args.seed,
                        "dtype": "float16", "score_indexing": "unchanged legacy multi-block baseline",
                        "use_cache": False},
        "reload_verified": False,
    }
    log_path = output_dir / "pruneme_log.json"
    with log_path.open("w", encoding="utf-8") as handle:
        json.dump(log, handle, indent=2, allow_nan=False)
    print("Reloading saved 2-bit checkpoint for verification...")
    reloaded = load_quantized_model(output_dir)
    with torch.no_grad():
        output = reloaded(input_ids=smoke_ids, attention_mask=smoke_mask,
                          output_hidden_states=True, use_cache=False)
        actual = output.logits[:, -1, :].float().cpu()
        if len(reloaded.model.layers) != len(kept) or len(output.hidden_states) != len(kept) + 1:
            raise ValueError("Saved model's layer count or hidden-state count is incorrect.")
        if not torch.isfinite(actual).all() or not torch.allclose(reference, actual, rtol=1e-3, atol=1e-3):
            raise ValueError("Reloaded logits disagree with the in-memory pruned model; do not use this output yet.")
        log["reload_max_logit_difference"] = (reference - actual).abs().max().item()
        del output
        generated = reloaded.generate(input_ids=smoke_ids, attention_mask=smoke_mask,
                                      max_new_tokens=16, do_sample=False)
        log["generation_smoke_test"] = tokenizer.decode(generated[0, smoke_ids.size(1):], skip_special_tokens=True)
    log["reload_verified"] = True
    with log_path.open("w", encoding="utf-8") as handle:
        json.dump(log, handle, indent=2, allow_nan=False)
    print(f"Verified saved model: {output_dir}")
    print(f"Layers: {num_layers} -> {len(kept)}; packed tensor size: {packed_bytes / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
