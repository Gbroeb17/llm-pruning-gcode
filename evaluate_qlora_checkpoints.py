"""Compare validation loss for PEFT/QLoRA adapter checkpoints.

This script only evaluates checkpoints. It does not train, merge, or modify them.
Its tokenization and completion-only label masking match qlora_heal.py.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_MODEL = PROJECT_DIR / "granite3b-pruneme-skip6-block18to23"
DEFAULT_VALID_JSONL = Path("/workspace/training_data/dataset_no_rule/valid.jsonl")
DEFAULT_CHECKPOINTS = [
    PROJECT_DIR / "qlora_runs/pruneme-skip6-qlora-ep1/checkpoint-1400",
    PROJECT_DIR / "qlora_runs/pruneme-skip6-qlora-ep1/checkpoint-1408",
]
DEFAULT_OUTPUT = (
    PROJECT_DIR
    / "qlora_runs/pruneme-skip6-qlora-ep1/checkpoint_eval_losses.json"
)


class CompletionOnlyCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            pad = max_len - len(feature["input_ids"])
            batch["input_ids"].append(
                feature["input_ids"] + [self.pad_id] * pad
            )
            batch["attention_mask"].append(
                feature["attention_mask"] + [0] * pad
            )
            batch["labels"].append(feature["labels"] + [-100] * pad)
        return {
            key: torch.tensor(value, dtype=torch.long)
            for key, value in batch.items()
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare eval_loss for QLoRA adapter checkpoints."
    )
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--valid-jsonl", type=Path, default=DEFAULT_VALID_JSONL)
    parser.add_argument(
        "--checkpoints",
        type=Path,
        nargs="+",
        default=DEFAULT_CHECKPOINTS,
    )
    parser.add_argument("--max-seq-length", type=int, default=12288)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def validate_paths(args: argparse.Namespace) -> None:
    missing = []
    if not args.base_model.is_dir():
        missing.append(f"Base model not found: {args.base_model}")
    if not args.valid_jsonl.is_file():
        missing.append(f"Validation JSONL not found: {args.valid_jsonl}")
    for checkpoint in args.checkpoints:
        if not (checkpoint / "adapter_config.json").is_file():
            missing.append(f"Invalid adapter checkpoint: {checkpoint}")
        if not (checkpoint / "adapter_model.safetensors").is_file():
            missing.append(f"Adapter weights not found: {checkpoint}")
    if missing:
        raise FileNotFoundError("\n".join(missing))


def build_validation_dataset(
    valid_jsonl: Path,
    tokenizer,
    max_seq_length: int,
) -> Dataset:
    examples = []
    malformed = 0
    fully_masked = 0

    with valid_jsonl.open(encoding="utf-8") as file:
        for line in file:
            try:
                messages = json.loads(line)["messages"]
                full_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                prefix_messages = [
                    message for message in messages if message["role"] != "assistant"
                ]
                prefix_text = tokenizer.apply_chat_template(
                    prefix_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                full_ids = tokenizer(
                    full_text,
                    truncation=True,
                    max_length=max_seq_length,
                    add_special_tokens=False,
                )["input_ids"]
                prefix_ids = tokenizer(
                    prefix_text,
                    add_special_tokens=False,
                )["input_ids"]

                labels = list(full_ids)
                for index in range(min(len(prefix_ids), len(full_ids))):
                    labels[index] = -100
                if not any(label != -100 for label in labels):
                    fully_masked += 1
                    continue

                examples.append(
                    {
                        "input_ids": full_ids,
                        "attention_mask": [1] * len(full_ids),
                        "labels": labels,
                    }
                )
            except Exception:
                malformed += 1

    if not examples:
        raise ValueError("Validation dataset has no usable samples.")
    print(
        f"Validation samples: {len(examples)} "
        f"(malformed={malformed}, fully_masked={fully_masked})"
    )
    return Dataset.from_list(examples)


def evaluate_checkpoint(
    *,
    base_model: Path,
    checkpoint: Path,
    dataset: Dataset,
    collator: CompletionOnlyCollator,
    batch_size: int,
) -> dict:
    print(f"\nEvaluating {checkpoint} ...")
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization,
        device_map="cuda:0",
    )
    model = PeftModel.from_pretrained(model, checkpoint, is_trainable=False)
    model.config.use_cache = False
    model.eval()

    eval_args = TrainingArguments(
        output_dir=str(PROJECT_DIR / ".checkpoint_eval_tmp"),
        per_device_eval_batch_size=batch_size,
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],
    )
    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=dataset,
        data_collator=collator,
    )
    metrics = trainer.evaluate()
    result = {
        "checkpoint": str(checkpoint.resolve()),
        "step": int(checkpoint.name.rsplit("-", 1)[-1]),
        "eval_loss": float(metrics["eval_loss"]),
        "eval_runtime": float(metrics.get("eval_runtime", 0.0)),
        "eval_samples_per_second": float(
            metrics.get("eval_samples_per_second", 0.0)
        ),
    }
    print(f"eval_loss={result['eval_loss']:.12g}")

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> int:
    args = parse_args()
    validate_paths(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for 4-bit QLoRA evaluation.")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dataset = build_validation_dataset(
        args.valid_jsonl,
        tokenizer,
        args.max_seq_length,
    )
    collator = CompletionOnlyCollator(tokenizer.pad_token_id)

    results = [
        evaluate_checkpoint(
            base_model=args.base_model,
            checkpoint=checkpoint,
            dataset=dataset,
            collator=collator,
            batch_size=args.batch_size,
        )
        for checkpoint in args.checkpoints
    ]
    results.sort(key=lambda result: result["eval_loss"])

    report = {
        "base_model": str(args.base_model.resolve()),
        "validation_jsonl": str(args.valid_jsonl.resolve()),
        "max_seq_length": args.max_seq_length,
        "num_validation_samples": len(dataset),
        "best_checkpoint": results[0]["checkpoint"],
        "best_eval_loss": results[0]["eval_loss"],
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\nRanking (lower eval_loss is better):")
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. checkpoint-{result['step']}: "
            f"eval_loss={result['eval_loss']:.12g}"
        )
    print(f"\nSaved report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
