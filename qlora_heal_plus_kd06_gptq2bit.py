"""Train LoRA adapters on the existing pruned kd06 GPTQ 2-bit checkpoint.

Keep the packed base frozen. Do not apply bitsandbytes NF4 or merge adapters
into GPTQ weights. Use the exact pruned base plus final_adapter for inference.
Run --smoke_test first to check adapter injection and backward without updates.
"""

import argparse
import json
import os
from pathlib import Path
import random
import shutil
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

PROJECT_DIR = Path("/media/ssd2/llm-pruning/cs_115522107/llm-pruning-gcode")
DEFAULT_BASE = PROJECT_DIR / "experiments/multi_block_pruning/outputs/kd06_seed44-gptq2bit-multiblock-b12-n1-skip12"
DATA_DIR = Path("/media/ssd2/llm-pruning/shared/datasets/training_data/dataset_no_rule")
RUN_NAME = "kd06_gptq2bit_skip12_ms16384_r16_ep2_run1"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model_dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--train_jsonl", type=Path, default=DATA_DIR / "train.jsonl")
    parser.add_argument("--valid_jsonl", type=Path, default=DATA_DIR / "valid.jsonl")
    parser.add_argument("--output_dir", type=Path, default=PROJECT_DIR / "qlora_runs" / RUN_NAME)
    parser.add_argument("--max_seq_length", type=int, default=16384)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--optim", choices=["adamw_torch", "paged_adamw_8bit"], default="adamw_torch")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=-1, help="Optional optimizer-step limit; otherwise use epochs.")
    parser.add_argument("--limit_train_samples", type=int)
    parser.add_argument("--limit_valid_samples", type=int)
    parser.add_argument("--resume_from_checkpoint", type=Path)
    parser.add_argument("--allow_nondeterministic", action="store_true",
                        help="Disable strict deterministic algorithms if an installed kernel requires it.")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Check one short forward/backward; no optimizer update or full dataset loading.")
    return parser.parse_args()


def validate_inputs(args):
    for name in ["max_seq_length", "lora_rank", "lora_alpha", "batch_size", "grad_accum",
                 "logging_steps", "eval_steps", "save_steps"]:
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive.")
    if args.max_seq_length < 2 or args.epochs <= 0 or args.lr <= 0:
        raise ValueError("Sequence length must be >= 2; epochs and lr must be positive.")
    if not 0 <= args.lora_dropout < 1 or not 0 <= args.warmup_ratio < 1:
        raise ValueError("Dropout and warmup ratio must be in [0, 1).")
    if args.max_steps == 0 or args.max_steps < -1:
        raise ValueError("max_steps must be positive or -1.")
    for limit in [args.limit_train_samples, args.limit_valid_samples]:
        if limit is not None and limit <= 0:
            raise ValueError("Sample limits must be positive or omitted.")
    with (args.base_model_dir / "config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    quantization = config.get("quantization_config", {})
    if quantization.get("quant_method") != "gptq" or quantization.get("bits") != 2:
        raise ValueError("Expected the existing GPTQ 2-bit base, not a floating-point/NF4 checkpoint.")
    if args.output_dir.resolve() == args.base_model_dir.resolve():
        raise ValueError("output_dir must not overwrite the base checkpoint.")
    if not args.smoke_test:
        if not args.train_jsonl.is_file() or not args.valid_jsonl.is_file():
            raise ValueError("Both train and held-out valid JSONL files must exist.")
        if args.train_jsonl.resolve() == args.valid_jsonl.resolve():
            raise ValueError("Train and validation must be separate files.")
        if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume_from_checkpoint:
            raise ValueError("output_dir is not empty. Choose a new run directory or explicitly resume a checkpoint.")
    return config


def load_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}, line {line_number}") from error
    return rows


def tokenize_sample(messages, tokenizer, max_seq_length):
    """Mask non-assistant tokens. Single-response chats follow the old recipe."""
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list.")
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    full_ids = tokenizer(full_text, truncation=True, max_length=max_seq_length,
                         add_special_tokens=False)["input_ids"]
    labels = [-100] * len(full_ids)
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prefix = tokenizer.apply_chat_template(messages[:index], tokenize=False, add_generation_prompt=True)
        through = tokenizer.apply_chat_template(messages[:index + 1], tokenize=False, add_generation_prompt=False)
        if not full_text.startswith(through) or not through.startswith(prefix):
            raise ValueError("Chat template is not prefix-consistent; cannot safely identify assistant tokens.")
        start = min(len(tokenizer(prefix, add_special_tokens=False)["input_ids"]), len(full_ids))
        end = min(len(tokenizer(through, add_special_tokens=False)["input_ids"]), len(full_ids))
        labels[start:end] = full_ids[start:end]
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


def build_dataset(rows, tokenizer, max_seq_length, split_name):
    from datasets import Dataset

    examples = []
    stats = {"raw_samples": len(rows), "kept_samples": 0, "no_assistant_tokens_after_truncation": 0,
             "malformed_samples": 0, "first_errors": []}
    for row_number, row in enumerate(rows, start=1):
        try:
            sample = tokenize_sample(row["messages"], tokenizer, max_seq_length)
            # Causal LM shifts labels by one, so index 0 alone is not supervision.
            if not any(label != -100 for label in sample["labels"][1:]):
                stats["no_assistant_tokens_after_truncation"] += 1
                continue
            examples.append(sample)
        except (KeyError, TypeError, ValueError) as error:
            stats["malformed_samples"] += 1
            if len(stats["first_errors"]) < 3:
                stats["first_errors"].append(f"row {row_number}: {error}")
    stats["kept_samples"] = len(examples)
    print(f"{split_name} tokenization: {stats}")
    if not examples:
        raise ValueError(f"No supervised samples remain in {split_name}; inspect templates and max_seq_length.")
    return Dataset.from_list(examples), stats


class CompletionOnlyCollator:
    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, features):
        import torch

        max_len = max(len(feature["input_ids"]) for feature in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            padding = max_len - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [self.pad_id] * padding)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * padding)
            batch["labels"].append(feature["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def load_lora_model(args):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, set_seed

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_dir, dtype=torch.float16, device_map={"": 0}, local_files_only=True,
    )
    model.config.use_cache = False
    projections = [module for name, module in model.named_modules() if name.rsplit(".", 1)[-1] in TARGET_MODULES]
    if not projections or any(not getattr(module, "SUPPORTS_TRAINING", False) for module in projections):
        raise ValueError("The loaded quantized projections must use a training-capable GPTQ backend (tested inference backend: TritonV2).")
    print(f"Base: {len(model.model.layers)} layers; projection backend: {type(projections[0])}")
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    set_seed(args.seed)
    config = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                        bias="none", task_type="CAUSAL_LM", target_modules=TARGET_MODULES)
    model = get_peft_model(model, config)
    model.peft_config["default"].base_model_name_or_path = str(args.base_model_dir.resolve())
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise ValueError("Expected only LoRA parameters to be trainable.")
    # FP16 AMP needs floating-point adapter/master parameters, not half gradients.
    for _, parameter in trainable:
        parameter.data = parameter.data.float()
    adapted_count = sum(hasattr(module, "lora_A") for module in model.modules())
    if adapted_count != len(projections):
        raise ValueError("PEFT did not attach adapters to every requested projection; inspect the PEFT version.")
    model.print_trainable_parameters()
    return model


def check_lora_gradients(model):
    import torch

    nonzero = []
    missing = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            if parameter.grad is not None:
                raise ValueError(f"Frozen base unexpectedly received gradients: {name}")
            continue
        if "lora_" not in name:
            raise ValueError(f"A non-adapter parameter is trainable: {name}")
        if parameter.grad is None:
            missing.append(name)
        elif not torch.isfinite(parameter.grad).all():
            raise ValueError(f"Non-finite LoRA gradients: {name}")
        elif torch.count_nonzero(parameter.grad).item():
            nonzero.append(name)
    if missing or not nonzero:
        raise ValueError(f"LoRA backward failed: missing gradients={missing[:3]}, nonzero tensors={len(nonzero)}")
    # LoRA A may correctly have zero gradients at initialization while B is zero.
    return {"nonzero_gradient_tensors": len(nonzero), "example_parameters": nonzero[:3]}


def backward_smoke_test(model, tokenizer):
    import torch

    sample = tokenize_sample(
        [{"role": "user", "content": "Generate a short G-code example."},
         {"role": "assistant", "content": "G21\nG90\nG0 X0 Y0\nG1 X10 Y10\n"}],
        tokenizer, 512,
    )
    if not any(label != -100 for label in sample["labels"][1:]):
        raise ValueError("Smoke-test chat contains no supervised assistant tokens.")
    batch = {key: value.to("cuda:0") for key, value in CompletionOnlyCollator(tokenizer)([sample]).items()}
    model.train()
    model.zero_grad(set_to_none=True)
    try:
        # Restore RNG afterward so the diagnostic does not change training dropout.
        with torch.random.fork_rng(devices=[0]):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise ValueError("Smoke-test loss is not finite.")
            loss.backward()
            result = dict(check_lora_gradients(model), loss=loss.item(), optimizer_updates=0)
        print(f"Backward smoke test PASSED: {result}")
        return result
    finally:
        model.zero_grad(set_to_none=True)


def main():
    args = parse_args()
    source_config = validate_inputs(args)

    import importlib.metadata
    import numpy as np
    import torch
    from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed

    if not torch.cuda.is_available():
        raise ValueError("Use the A100 pruning-gptq CUDA environment.")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This script is for one GPU; do not launch it with torchrun/DDP.")
    started = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = not args.allow_nondeterministic
    torch.use_deterministic_algorithms(not args.allow_nondeterministic)
    torch.cuda.reset_peak_memory_stats(0)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_dir, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer needs an EOS or padding token.")
    tokenizer.padding_side = "right"
    model = load_lora_model(args)
    smoke_result = backward_smoke_test(model, tokenizer)
    if args.smoke_test:
        print("Smoke test complete. No weights were updated and no adapter was saved.")
        return
    train_rows = load_jsonl(args.train_jsonl)
    valid_rows = load_jsonl(args.valid_jsonl)
    random.Random(args.seed).shuffle(train_rows)
    if args.limit_train_samples is not None:
        train_rows = train_rows[:args.limit_train_samples]
    if args.limit_valid_samples is not None:
        valid_rows = valid_rows[:args.limit_valid_samples]
    train_dataset, train_stats = build_dataset(train_rows, tokenizer, args.max_seq_length, "train")
    valid_dataset, valid_stats = build_dataset(valid_rows, tokenizer, args.max_seq_length, "valid")
    training_args = TrainingArguments(
        output_dir=str(args.output_dir), num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, prediction_loss_only=True,
        learning_rate=args.lr, warmup_ratio=args.warmup_ratio, lr_scheduler_type="cosine",
        optim=args.optim, fp16=True, bf16=False, max_grad_norm=0.3,
        logging_steps=args.logging_steps, eval_strategy="steps", eval_steps=args.eval_steps,
        save_strategy="steps", save_steps=args.save_steps, save_total_limit=2,
        report_to="none", seed=args.seed, data_seed=args.seed,
        full_determinism=not args.allow_nondeterministic, remove_unused_columns=False,
        dataloader_num_workers=0, label_names=["labels"],
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset,
                      eval_dataset=valid_dataset, data_collator=CompletionOnlyCollator(tokenizer))
    print("Starting GPTQ 2-bit + LoRA healing (adapter only, no NF4 conversion or merge).")
    result = trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
    metrics = dict(result.metrics)
    metrics.update(trainer.evaluate())
    adapter_dir = args.output_dir / "final_adapter"
    trainer.model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    trainer.save_state()
    source_log = args.base_model_dir / "pruneme_log.json"
    if source_log.is_file():
        shutil.copy2(source_log, adapter_dir / "base_pruneme_log.json")
    versions = {}
    for package in ["torch", "transformers", "peft", "gptqmodel", "optimum", "datasets"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    torch.cuda.synchronize()
    report = {
        "method": "LoRA on frozen GPTQ 2-bit pruned base",
        "base_model_dir": str(args.base_model_dir.resolve()), "adapter_dir": str(adapter_dir.resolve()),
        "merged": False, "requires_original_pruned_base_for_inference": True,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "quantization_config": source_config["quantization_config"], "versions": versions,
        "backward_smoke_test": smoke_result, "train_data": train_stats, "valid_data": valid_stats,
        "metrics": metrics, "history": trainer.state.log_history,
        "elapsed_seconds": time.time() - started,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(0) / 1024**3,
    }
    with (args.output_dir / "gptq_lora_training_log.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Adapter saved: {adapter_dir}")
    print(f"Keep this exact base checkpoint for inference: {args.base_model_dir}")
    print("No merged model was produced; load the base with PeftModel.from_pretrained(base, adapter_dir).")


if __name__ == "__main__":
    main()
