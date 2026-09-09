import math

TRAIN_JSONL = "/workspace/training_data/dataset_no_rule/train.jsonl"

BATCH_SIZE = 1
GRAD_ACCUM = 8

EPOCHS_TO_CHECK = [0.125, 0.25, 0.5, 1.0, 2.0]


# Count JSONL samples
with open(TRAIN_JSONL, "r", encoding="utf-8") as f:
    num_samples = sum(1 for line in f if line.strip())

effective_batch_size = BATCH_SIZE * GRAD_ACCUM

steps_per_epoch = math.ceil(
    num_samples / effective_batch_size
)

print(f"Training samples     : {num_samples}")
print(f"Batch size           : {BATCH_SIZE}")
print(f"Gradient accumulation: {GRAD_ACCUM}")
print(f"Effective batch size : {effective_batch_size}")
print(f"Steps / epoch        : {steps_per_epoch}")
print()

for epochs in EPOCHS_TO_CHECK:
    total_steps = math.ceil(steps_per_epoch * epochs)
    print(f"{epochs:>5} epoch -> ~{total_steps} optimizer steps")