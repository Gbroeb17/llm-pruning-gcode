
import json
from collections import Counter
from transformers import AutoTokenizer

BASE_MODEL_DIR = "/media/ssd2/llm-pruning/cs_115522107/llm-pruning-gcode/granite3b-pruneme-skip16-block14to29"

TRAIN_JSONL = "/media/ssd2/llm-pruning/shared/datasets/training_data/dataset_no_rule/train.jsonl"
VALID_JSONL = "/media/ssd2/llm-pruning/shared/datasets/training_data/dataset_no_rule/valid.jsonl"

MAX_SEQ_LENGTH = 12288

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)


def analyze_dataset(path, name):
    lengths = []
    over_limit = []
    eos_missing_after_truncation = 0
    ends_with_eos = 0

    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            row = json.loads(line)

            full_text = tokenizer.apply_chat_template(
                row["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )

            token_ids = tokenizer(
                full_text,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]

            length = len(token_ids)
            lengths.append(length)

            if length > MAX_SEQ_LENGTH:
                over_limit.append((idx, length))

            if token_ids and token_ids[-1] == tokenizer.eos_token_id:
                ends_with_eos += 1
                truncated_ids = token_ids[:MAX_SEQ_LENGTH]
                if truncated_ids[-1] != tokenizer.eos_token_id:
                    eos_missing_after_truncation += 1

    total = len(lengths)

    print(f"\n===== {name} =====")
    print(f"Total samples: {total}")
    print(f"Min tokens: {min(lengths)}")
    print(f"Max tokens: {max(lengths)}")
    print(f"Average tokens: {sum(lengths) / total:.1f}")
    print(f"Over {MAX_SEQ_LENGTH}: {len(over_limit)}")
    print(f"Over-limit ratio: {len(over_limit) / total * 100:.2f}%")
    print(f"Sequences ending with tokenizer EOS: {ends_with_eos}")
    print(f"EOS lost after truncation: {eos_missing_after_truncation}")

    print("\nTop 10 longest samples:")
    for idx, length in sorted(
        enumerate(lengths),
        key=lambda x: x[1],
        reverse=True
    )[:10]:
        print(f"  Sample {idx}: {length} tokens")

    return lengths


train_lengths = analyze_dataset(TRAIN_JSONL, "TRAIN")
valid_lengths = analyze_dataset(VALID_JSONL, "VALID")


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)

with open(TRAIN_JSONL, encoding="utf-8") as f:
    row = json.loads(next(f))

full_text = tokenizer.apply_chat_template(
    row["messages"],
    tokenize=False,
    add_generation_prompt=False,
)

ids = tokenizer(
    full_text,
    add_special_tokens=False,
)["input_ids"]

print("Tokenizer EOS:", repr(tokenizer.eos_token))
print("EOS ID:", tokenizer.eos_token_id)
print("Last 150 characters:", repr(full_text[-150:]))
print("Last 15 tokens:", tokenizer.convert_ids_to_tokens(ids[-15:]))
print("Last token ID:", ids[-1])
print("Special tokens:", tokenizer.special_tokens_map)

from transformers import AutoTokenizer, AutoConfig, GenerationConfig

model_dir = BASE_MODEL_DIR

tokenizer = AutoTokenizer.from_pretrained(model_dir)
config = AutoConfig.from_pretrained(model_dir)

print("Tokenizer EOS:", tokenizer.eos_token_id)
print("Model EOS:", config.eos_token_id)

try:
    gen_config = GenerationConfig.from_pretrained(model_dir)
    print("Generation EOS:", gen_config.eos_token_id)
except Exception as e:
    print("No generation config:", e)

print(repr(row["messages"][-1]["content"][-100:]))
print(repr(full_text[-100:]))
print(repr(tokenizer.chat_template[-500:]))