"""
Streamline-style healing for an already-pruned Granite model.

This script does not do QLoRA and does not prune layers.
It trains one decoder layer in the pruned model with MSE loss so that it
imitates the hidden-state transformation of the removed layer block in the
original full Granite model.

Required:
  1. original_model_dir: the full, unpruned Granite model
  2. pruned_model_dir: the already-pruned Granite model
  3. removed_start_layer: first original layer that was removed
  4. removed_count: number of consecutive original layers removed

If original layers 18 to 23 were removed, use:
  --removed_start_layer 18 --removed_count 6

The replacement layer index is removed_start_layer - 1, matching the
LLM-Streamline idea:
  original layer 17 is trained to behave like original layers 17..23.
After training, that healed layer replaces layer 17 in the pruned model.
"""

'''
調參可能要改max_seq_len
抽limit_sample筆資料訓練, 目前沒設不知道全部資料training會不會很久
之後可以試試看和qlora結合
'''

import argparse
import copy
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ======== same as pruneme setup ====================
class JsonlGCodeDataset(Dataset):
    def __init__(
        self,
        jsonl_path,
        tokenizer,
        max_seq_len,
        limit_samples=None,   # 抽幾筆資料拿來訓練, none就是選全部資料
        min_seq_len=1024,
        short_threshold=3600,
        long_threshold=20000,
        short_ratio=0.5,
        medium_ratio=0.3,
        long_ratio=0.2,
        seed=42,
    ):
        self.max_seq_len = max_seq_len
        self.rng = random.Random(seed)
        self.samples = []
        raw = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                row = json.loads(line)
                text = tokenizer.apply_chat_template(
                    row["messages"],
                    tokenize=False,
                    add_generation_prompt=False,
                )
                ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                raw.append({"input_ids": ids, "tok_len": len(ids)})

        pool = [r for r in raw if r["tok_len"] >= min_seq_len]
        short = [r for r in pool if r["tok_len"] < short_threshold]
        medium = [r for r in pool if short_threshold <= r["tok_len"] < long_threshold]
        long = [r for r in pool if r["tok_len"] >= long_threshold]
        print(f"Tokenized raw samples: {len(raw)}")
        print(f"Pool >= {min_seq_len} tokens: {len(pool)}")
        print(f"Buckets short/medium/long: {len(short)}/{len(medium)}/{len(long)}")

        if limit_samples:
            chosen = (
                self._pick(short, int(limit_samples * short_ratio))
                + self._pick(medium, int(limit_samples * medium_ratio))
                + self._pick(long, int(limit_samples * long_ratio))
            )

            while len(chosen) < limit_samples:
                chosen_ids = {id(r) for r in chosen}
                leftover = [r for r in pool if id(r) not in chosen_ids]
                if not leftover:
                    break
                chosen.append(self.rng.choice(leftover))
        else:
            chosen = list(pool)

        self.rng.shuffle(chosen)
        self.samples = [r["input_ids"] for r in chosen[:limit_samples]]
        print(f"Selected healing samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    # 每次長 sample 每次被取到時，會隨機選一段長度為max_seq_len的chunk長度來訓練
    def __getitem__(self, index):
        ids = self.samples[index]
        if len(ids) > self.max_seq_len:
            start = self.rng.randint(0, len(ids) - self.max_seq_len)
            ids = ids[start : start + self.max_seq_len]
        return torch.tensor(ids, dtype=torch.long)

    def _pick(self, bucket, n):
        return self.rng.sample(bucket, min(n, len(bucket)))


# 幫 DataLoader 把不同長度的 token 序列補齊成同一長度，並建立 attention_mask
class PadCollator:
    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, samples):
        max_len = max(x.numel() for x in samples)
        input_ids = []
        attention_mask = []

        for ids in samples:
            pad_len = max_len - ids.numel()
            input_ids.append(torch.cat([ids, torch.full((pad_len,), self.pad_id, dtype=torch.long)]))
            attention_mask.append(torch.cat([torch.ones_like(ids), torch.zeros(pad_len, dtype=torch.long)]))

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
        }


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError("Expected a Granite/LLaMA-style model with model.model.layers.")


def set_layers(model, layers):
    model.model.layers = nn.ModuleList(layers)
    model.config.num_hidden_layers = len(layers)
    model.model.config.num_hidden_layers = len(layers)


def make_tokenizer(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer

#
#@torch.no_grad()
#def teacher_hidden_pair(teacher, batch, replacement_layer_idx, removed_count):
#    teacher.eval()
#    out = teacher(**batch, output_hidden_states=True, use_cache=False)

    # hidden_states[0] is embedding output.
    # hidden_states[i] is input to decoder layer i.
    # hidden_states[i + 1] is output of decoder layer i.
#    x = out.hidden_states[replacement_layer_idx]
#    y = out.hidden_states[replacement_layer_idx + removed_count + 1]
#    return x.detach(), y.detach()
#

# 用forward hook只抓指定輸出層的hidden state
@torch.no_grad()
def teacher_hidden_pair(teacher, batch, replacement_layer_idx, removed_count):
    layers = get_layers(teacher)
    target_output_layer_idx = replacement_layer_idx + removed_count
    cache = {}

    def save_input(module, inputs):
        cache["x"] = inputs[0].detach()

    def save_output(module, inputs, output):
        if isinstance(output, tuple):
            output = output[0]
        cache["y"] = output.detach()

    # teacher forward 時，一進入 replace layer 前，就會自動執行 save_input()，存下 replace layer input
    input_handle = layers[replacement_layer_idx].register_forward_pre_hook(save_input)
    output_handle = layers[target_output_layer_idx].register_forward_hook(save_output)

    teacher.eval()
    teacher(**batch, use_cache=False)

    input_handle.remove()
    output_handle.remove()

    return cache["x"], cache["y"]

def run_layer(layer, hidden_states, attention_mask):
    seq_len = hidden_states.size(1)
    position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(hidden_states.size(0), -1)

    try:
        out = layer(
            hidden_states=hidden_states,
            attention_mask=None,
            position_ids=position_ids,
        )
    except TypeError:
        out = layer(
            hidden_states=hidden_states,
            position_ids=position_ids,
        )

    return out[0] if isinstance(out, tuple) else out


def train_streamline_replacement(
    teacher,
    pruned,
    dataloader,
    device,
    replacement_layer_idx,
    removed_count,
    epochs,
    lr,
    min_lr,
    warmup_ratio,
    cosine_training_ratio,
    weight_decay,
    grad_accum,
):
    pruned_layers = get_layers(pruned)
    replacement = copy.deepcopy(pruned_layers[replacement_layer_idx]).to(device)
    replacement.train()

    for p in teacher.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(replacement.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    total_optimizer_steps = max(1, (len(dataloader) * epochs) // grad_accum)
    scheduler_steps = max(1, int(total_optimizer_steps * cosine_training_ratio))
    warmup_steps = max(1, int(scheduler_steps * warmup_ratio))

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)

        progress = min(
            1.0,
            float(current_step - warmup_steps) / float(max(1, scheduler_steps - warmup_steps)),
        )
        cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi))).item()
        min_lr_ratio = min_lr / lr
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_loss = float("inf")
    best_state = copy.deepcopy(replacement.state_dict())
    optimizer_step = 0

    for epoch in range(epochs):
        total_loss = 0.0
        total_steps = 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(tqdm(dataloader, desc=f"Healing epoch {epoch + 1}")):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.no_grad():
                teacher_x, teacher_y = teacher_hidden_pair(
                    teacher,
                    batch,
                    replacement_layer_idx,
                    removed_count,
                )

            pred = run_layer(replacement, teacher_x, batch["attention_mask"])
            mask = batch["attention_mask"].bool().unsqueeze(-1).expand_as(pred)
            loss = loss_fn(pred.float().masked_select(mask), teacher_y.float().masked_select(mask))
            loss = loss / grad_accum
            loss.backward()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(replacement.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

            total_loss += loss.item() * grad_accum
            total_steps += 1

        avg_loss = total_loss / max(total_steps, 1)
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch + 1}: train_mse={avg_loss:.6f}, lr={current_lr:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = copy.deepcopy(replacement.state_dict())

    replacement.load_state_dict(best_state)
    return replacement.cpu()


def save_healed_pruned_model(pruned, healed_layer, replacement_layer_idx, output_dir, tokenizer):
    layers = list(get_layers(pruned))
    layers[replacement_layer_idx] = healed_layer
    pruned = pruned.cpu()
    set_layers(pruned, layers)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    pruned.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_model_dir", required=True)
    parser.add_argument("--pruned_model_dir", required=True)
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--removed_start_layer", type=int, required=True)
    parser.add_argument("--removed_count", type=int, required=True)

    parser.add_argument("--max_seq_len", type=int, default=2048)  # 每次真正丟進模型訓練的token長度上限
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
    parser.add_argument("--cosine_training_ratio", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.removed_start_layer <= 0:
        raise ValueError("removed_start_layer must be >= 1 because the previous layer is used as replacement.")

    replacement_layer_idx = args.removed_start_layer - 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    tokenizer = make_tokenizer(args.original_model_dir)

    # ================ trainging data setup =======================
    dataset = JsonlGCodeDataset(
        args.train_jsonl,
        tokenizer,
        args.max_seq_len,
        args.limit_samples,
        min_seq_len=args.min_seq_len,
        short_threshold=args.short_threshold,
        long_threshold=args.long_threshold,
        short_ratio=args.short_ratio,
        medium_ratio=args.medium_ratio,
        long_ratio=args.long_ratio,
        seed=args.seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=PadCollator(tokenizer),
    )

    # ===================== loading teacher model and already-pruned model ===================================
    print(f"Loading original teacher model from: {args.original_model_dir}")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.original_model_dir,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=True,
    ).to(device)
    teacher.config.use_cache = False

    print(f"Loading already-pruned model from: {args.pruned_model_dir}")
    pruned = AutoModelForCausalLM.from_pretrained(
        args.pruned_model_dir,
        torch_dtype=dtype,
        device_map=None,
        trust_remote_code=True,
    ).to(device)
    pruned.config.use_cache = False

    # ========================= train replaced layer =============================================
    original_layers = len(get_layers(teacher))
    pruned_layers = len(get_layers(pruned))
    expected_pruned_layers = original_layers - args.removed_count

    print(f"Original layers: {original_layers}")
    print(f"Pruned layers: {pruned_layers}")
    print(f"Replacement layer index in pruned model: {replacement_layer_idx}")

    if pruned_layers != expected_pruned_layers:
        print(
            "Warning: pruned layer count does not equal "
            f"original_layers - removed_count ({expected_pruned_layers}). "
            "Continue only if your pruned checkpoint has extra architecture changes."
        )

    healed_layer = train_streamline_replacement(
        teacher=teacher,
        pruned=pruned,
        dataloader=dataloader,
        device=device,
        replacement_layer_idx=replacement_layer_idx,
        removed_count=args.removed_count,
        epochs=args.epochs,
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_ratio=args.warmup_ratio,
        cosine_training_ratio=args.cosine_training_ratio,
        weight_decay=args.weight_decay,
        grad_accum=args.grad_accum,
    )

    # ================================ merge trained replaced layer and pruned model =============================================
    save_healed_pruned_model(
        pruned=pruned,
        healed_layer=healed_layer,
        replacement_layer_idx=replacement_layer_idx,
        output_dir=args.output_dir,
        tokenizer=tokenizer,
    )

    print(f"Saved Streamline-healed pruned Granite model to: {args.output_dir}")


if __name__ == "__main__":
    main()
