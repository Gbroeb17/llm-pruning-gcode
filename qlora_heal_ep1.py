"""
QLoRA healing fine-tune for PruneMe-pruned Granite-3B on G-code task.

Purpose:
  After PruneMe removes 6 layers, the model loses ability to produce
  precise G-code format (degeneration / repeat / missing Y / wrong G0/G1).
  This script applies QLoRA fine-tune on the original G-code training set
  to recover task-specific format generation.

Pattern follows HF PEFT QLoRA standard recipe:
  https://huggingface.co/docs/peft (LoraConfig + prepare_model_for_kbit_training)
  https://github.com/huggingface/trl  (SFTTrainer not used; we use plain Trainer
                                       with custom collator for completion-only loss)

After training:
  1. LoRA adapter is merged back to base weights
  2. Final model saved as standard transformers checkpoint
  3. Distillation teammate can load it normally

Run:
  CUDA_VISIBLE_DEVICES=2 python qlora_heal.py
"""

import json
import os
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


# ============== config ==============
# Path to pruned model produced by pruneme_granite.py
BASE_MODEL_DIR = "/home/cs_114522026/mywork/llm-compressor/granite3b-pruneme-skip6-block18to23"
# Adjust to your actual block range — check the folder name you got

TRAIN_JSONL = "/home/cs_114522026/mywork/llm-compressor/final_full_dataset/training_data/dataset_no_rule/train.jsonl"
VALID_JSONL = "   " #等輸入
VALID_SPLIT_RATIO = 0.05   # 5% from train.jsonl as validation

OUTPUT_DIR     = "qlora_runs/pruneme_heal_ep2_full"
MERGED_SAVE_DIR = "granite3b-pruneme-healed-ep2"

# Training hyperparameters (sensible defaults, adjust as needed)
MAX_SEQ_LENGTH      = 12288
LORA_RANK           = 16
LORA_ALPHA          = 32
LORA_DROPOUT        = 0.05
LEARNING_RATE       = 1e-4
NUM_EPOCHS          = 2.0
BATCH_SIZE          = 1
GRAD_ACCUM          = 8        # effective batch size = 8
WARMUP_RATIO        = 0.03
LOG_STEPS           = 10
EVAL_STEPS          = 200
SAVE_STEPS          = 100

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)


# ============== load tokenizer ==============
print(f"Loading tokenizer from {BASE_MODEL_DIR}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"   # important: right-pad for causal LM


# ============== load model with 4-bit quantization (QLoRA) ==============
# Standard QLoRA pattern from HF docs:
# 4-bit nf4 quantization + bf16 compute for fast training on small GPUs.
# Even though A100 80GB has plenty of VRAM, we follow QLoRA paper pattern
# for reproducibility. To use plain LoRA (no quantize), set USE_QLORA=False.

USE_QLORA = True

if USE_QLORA:
    print("Loading base model with 4-bit quantization (QLoRA)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR,
        quantization_config=bnb_config,
        device_map="cuda:0",
    )
    model = prepare_model_for_kbit_training(model)
else:
    print("Loading base model in bf16 (plain LoRA)...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

# Required for training: tell model not to use cache (incompatible with gradient checkpointing)
model.config.use_cache = False


# ============== apply LoRA adapter ==============
# Target all linear layers in attention + MLP (standard LLaMA-family target)
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# ============== load training data and valid data ==============
print(f"\nLoading training data from {TRAIN_JSONL}")
print(f"\nLoading training data from {VALID_JSONL}")

def load_jsonl(path: str) -> list:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

all_rows_train = load_jsonl(TRAIN_JSONL)
all_rows_valid  = load_jsonl(VALID_JSONL)
random.shuffle(all_rows_train)
#n_valid = int(len(all_rows_train) * VALID_SPLIT_RATIO)
#valid_rows = all_rows_train[:n_valid]
#train_rows = all_rows_train[n_valid:]
#print(f"  total: {len(all_rows_train)} samples")
print(f"  train: {len(all_rows_train)} samples")
print(f"  valid: {len(all_rows_valid)} samples (held out from train)")

# Filter out samples that are too long after tokenization
# (avoids wasting compute on samples that will be heavily truncated)
def quick_filter_by_length(rows, max_chars=15000):
    """Approximate filter: ~3 chars per token, so 15000 chars ~ 5000 tokens.
    Removes the longest few % to keep training efficient at MAX_SEQ_LENGTH=4096.
    Returns filtered list and stats."""
    filtered = []
    for r in rows:
        total_chars = sum(len(m.get("content", "")) for m in r["messages"])
        if total_chars <= max_chars:
            filtered.append(r)
    return filtered

# Optional: limit very long samples (uncomment to enable)
# train_rows = quick_filter_by_length(train_rows, max_chars=15000)
# print(f"  train after length filter: {len(train_rows)} samples")


# ============== tokenization with completion-only loss ==============
# For "healing" fine-tune, we want loss ONLY on the assistant response
# (the G-code part), not on the system/user prompt. This teaches the
# model to generate good outputs without wasting training signal on
# reconstructing inputs.
#
# How: tokenize the full chat, then find where "Answer:" appears
# (Granite's chat template uses "Answer:" as assistant prefix), and
# mask everything before that to -100 in labels.

# Detect the assistant prefix that Granite's chat_template uses.
# We compute it once by templating a dummy conversation.
dummy_messages = [
    {"role": "system",    "content": "S"},
    {"role": "user",      "content": "U"},
    {"role": "assistant", "content": "A"},
]
dummy_text = tokenizer.apply_chat_template(
    dummy_messages, tokenize=False, add_generation_prompt=False,
)
# Find the prefix that comes right before the assistant content "A"
assistant_idx = dummy_text.rfind("A")
ASSISTANT_PREFIX = dummy_text[:assistant_idx]   # "...Answer:\n"
print(f"\nDetected assistant prefix: {repr(ASSISTANT_PREFIX[-30:])}")


def tokenize_sample(messages: list) -> dict:
    """Tokenize one conversation, mask non-assistant tokens with -100."""
    # Full text including assistant response
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )
    # Text up to (but not including) the assistant response
    prefix_messages = [m for m in messages if m["role"] != "assistant"]
    prefix_text = tokenizer.apply_chat_template(
        prefix_messages, tokenize=False, add_generation_prompt=True,
    )
    
    # Tokenize both; the prefix length tells us where response starts
    full_ids = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        add_special_tokens=False,
    )["input_ids"]
    prefix_ids = tokenizer(
        prefix_text,
        add_special_tokens=False,
    )["input_ids"]
    
    # Build labels: -100 for prefix, real token ID for response
    labels = list(full_ids)
    n_prefix = min(len(prefix_ids), len(full_ids))
    for i in range(n_prefix):
        labels[i] = -100
    
    return {
        "input_ids":      full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels":         labels,
    }


def build_dataset(rows: list) -> Dataset:
    examples = []
    for r in rows:
        try:
            ex = tokenize_sample(r["messages"])
            # Skip samples where all labels are masked (would give 0 loss)
            if any(l != -100 for l in ex["labels"]):
                examples.append(ex)
        except Exception as e:
            continue   # skip malformed
    return Dataset.from_list(examples)

print("\nTokenizing train + valid...")
train_dataset = build_dataset(all_rows_train)
valid_dataset = build_dataset(all_rows_valid)
print(f"  train tokenized: {len(train_dataset)}")
print(f"  valid tokenized: {len(valid_dataset)}")


# ============== data collator ==============
# Custom collator: pad input_ids/attention_mask/labels to the longest in batch.
# We can't use DataCollatorForLanguageModeling because it doesn't preserve
# our -100 mask on labels.
class CompletionOnlyCollator:
    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id
    
    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            n = len(f["input_ids"])
            pad = max_len - n
            batch["input_ids"].append(f["input_ids"] + [self.pad_id] * pad)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad)
            batch["labels"].append(f["labels"] + [-100] * pad)   # padding ignored in loss
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}

collator = CompletionOnlyCollator(tokenizer)


# ============== training arguments ==============
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    gradient_checkpointing=True,
    
    learning_rate=LEARNING_RATE,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type="cosine",
    
    optim="paged_adamw_8bit" if USE_QLORA else "adamw_torch",
    bf16=True,
    max_grad_norm=0.3,
    
    logging_steps=LOG_STEPS,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=2,
    
    report_to="none",   # don't push to wandb/tensorboard unless you want to
    seed=SEED,
    
    # Don't remove unused columns — our dataset only has input_ids/labels/attention_mask
    remove_unused_columns=False,
)


# ============== train ==============
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=collator,
)

print("\n=== Starting QLoRA training ===")
trainer.train()
print("Training complete.")


# ============== merge LoRA and save final model ==============
print(f"\nMerging LoRA adapter into base weights...")

# Save adapter-only checkpoint first (for re-loading if needed)
adapter_dir = f"{OUTPUT_DIR}/final_adapter"
trainer.model.save_pretrained(adapter_dir)
tokenizer.save_pretrained(adapter_dir)
print(f"  Adapter saved to: {adapter_dir}")

# Now merge for clean handoff to distillation teammate
# Re-load in bf16 (not 4-bit) because merge_and_unload doesn't work on quantized model
del model, trainer
torch.cuda.empty_cache()

from peft import PeftModel

print("Reloading base in bf16 for merge...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_DIR,
    dtype=torch.bfloat16,
    device_map="cuda:0",
)
peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
merged_model = peft_model.merge_and_unload()

print(f"Saving merged model to {MERGED_SAVE_DIR}/")
Path(MERGED_SAVE_DIR).mkdir(parents=True, exist_ok=True)
merged_model.save_pretrained(MERGED_SAVE_DIR, safe_serialization=True)
tokenizer.save_pretrained(MERGED_SAVE_DIR)

# Final size
total_bytes = sum(
    os.path.getsize(os.path.join(MERGED_SAVE_DIR, f))
    for f in os.listdir(MERGED_SAVE_DIR)
    if os.path.isfile(os.path.join(MERGED_SAVE_DIR, f))
)
print(f"\n✓ Healed model saved: {MERGED_SAVE_DIR}/")
print(f"  Size: {total_bytes / 1e9:.2f} GB")
print(f"\nFor inference, point your inference script to:")
print(f"  MODEL_OR_ADAPTER_DIR = '{MERGED_SAVE_DIR}'")
print("Done.")
