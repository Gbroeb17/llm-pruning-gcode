"""
Layer pruning for Granite-Code-3B on G-code task.

Implementation faithful to:
  - Paper: "The Unreasonable Ineffectiveness of the Deeper Layers"
           (Gromov et al., ICLR 2025)
  - Code:  arcee-ai/PruneMe (unofficial implementation)

Algorithm (from PruneMe's layer_similarity.py + utils.py):

  1. Load model with output_hidden_states=True
     → one forward pass gives ALL layers' hidden states

  2. For each calibration sample, extract the LAST NON-PADDED token's
     hidden state at every layer (paper §3.2, formula 7 footnote:
     "we chose to focus on the final token since, due to the causal
     attention mask, its embedding is the only one that depends on
     the entire sequence")

  3. For each possible starting layer l (0..L-N):
     compute angular_distance(h[l], h[l+N]):
       d = (1/π) * arccos( cos_sim(h[l], h[l+N]) )
     Average over all calibration samples.

  4. Find l* = argmin_l d  →  best connected block to prune

  5. Remove layers l* to l*+N-1 from model.model.layers

  6. (Optional) QLoRA healing fine-tune (not in this script)

Adaptation for Granite-3B:
  - Calibration: G-code train.jsonl with chat template (not generic text)
  - Stratified sampling: same as SparseGPT setup (short/medium/long buckets)
  - Untie lm_head before save (Granite uses tied embeddings)
  - No BitsAndBytes quantization (A100 80GB has enough VRAM)

Run:
  CUDA_VISIBLE_DEVICES=2 python pruneme_granite.py
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============== config ==============
MODEL_ID  = "/workspace/models/granite_gcode_merged/granite_gcode_merged_best_new"
DATA_PATH = "/workspace/training_data/dataset_no_rule/train.jsonl"

NUM_CALIBRATION_SAMPLES = 128
MAX_SEQ_LENGTH          = 65536
LAYERS_TO_SKIP          = 6          # number of consecutive layers to remove (= block size)
                                     # Granite has 32 layers; 6 = 18.75% pruning
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============== angular distance (faithful to PruneMe utils.py) ==============
def angular_distance(x_l: torch.Tensor, x_l_plus_n: torch.Tensor) -> torch.Tensor:
    """Compute angular distance between two sets of hidden states.
    
    Faithful to PruneMe/compute_block_similarity/utils.py line 4-9.
    
    Args:
        x_l:         (B, H) last-token hidden states at layer l
        x_l_plus_n:  (B, H) last-token hidden states at layer l+n
    
    Returns:
        (B,) angular distances in [0, 1]
    """
    x_l_norm = x_l / torch.norm(x_l, dim=-1, keepdim=True)
    x_l_plus_n_norm = x_l_plus_n / torch.norm(x_l_plus_n, dim=-1, keepdim=True)
    cosine_similarity = (x_l_norm * x_l_plus_n_norm).sum(-1)
    return torch.acos(cosine_similarity.clamp(min=-1, max=1)) / torch.pi


def compute_block_distances(hidden_states, layers_to_skip: int):
    """Compute angular distance for each possible block starting position.
    
    Faithful to PruneMe/compute_block_similarity/utils.py line 11-18.
    
    Args:
        hidden_states: list of (B, H) tensors, one per layer (L elements)
        layers_to_skip: block size N
    
    Returns:
        list of float, one distance per starting position
    """
    distances = []
    num_layers = len(hidden_states)
    for l in range(num_layers - layers_to_skip):
        block_distance = angular_distance(
            hidden_states[l], hidden_states[l + layers_to_skip]
        ).mean().item()
        distances.append(block_distance)
    return distances


def get_last_non_padded_tokens(hidden_states, attention_mask):
    """Extract last non-padded token's hidden state for each layer.
    
    Faithful to PruneMe/compute_block_similarity/utils.py line 20-31.
    
    Args:
        hidden_states: tuple of (L+1) tensors, each (B, T, H)
                       (from model output with output_hidden_states=True)
        attention_mask: (B, T) tensor
    
    Returns:
        list of (B, H) tensors, one per layer
    """
    last_non_padded = []
    for layer_hidden in hidden_states:
        batch_size = layer_hidden.size(0)
        batch_last_tokens = []
        for b in range(batch_size):
            # Find last non-padded position using attention_mask
            last_non_pad_idx = attention_mask[b].nonzero(as_tuple=True)[0].max()
            last_token = layer_hidden[b, last_non_pad_idx, :]   # (H,)
            batch_last_tokens.append(last_token.unsqueeze(0))   # (1, H)
        last_non_padded.append(torch.cat(batch_last_tokens, dim=0))  # (B, H)
    return last_non_padded


# ============== load model + tokenizer ==============
print(f"Loading model from {MODEL_ID}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="cuda:0",
    low_cpu_mem_usage=True,
    output_hidden_states=True,      # ← PruneMe's approach: get all hidden states
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# PruneMe sets pad_token = eos_token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

num_layers = model.config.num_hidden_layers
print(f"Model: {num_layers} layers, hidden={model.config.hidden_size}")
print(f"Plan: find best block of {LAYERS_TO_SKIP} consecutive layers to remove")


# ============== build calibration data ==============
print("\nBuilding calibration data...")

raw = []
with open(DATA_PATH, encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        text = tokenizer.apply_chat_template(
            d["messages"], tokenize=False, add_generation_prompt=False,
        )
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        raw.append({"input_ids": ids, "tok_len": len(ids)})

# Stratified sampling (same as SparseGPT/ShortGPT setup)
MIN_LEN = 1024
pool = [r for r in raw if r["tok_len"] >= MIN_LEN]
rng = random.Random(SEED)
short  = [r for r in pool if r["tok_len"] < 3600]
medium = [r for r in pool if 3600 <= r["tok_len"] < 20000]
long_  = [r for r in pool if r["tok_len"] >= 20000]
print(f"Pool: {len(pool)}, buckets: {len(short)}/{len(medium)}/{len(long_)}")

def pick(bucket, n):
    return rng.sample(bucket, min(n, len(bucket)))

chosen = (
    pick(short,  int(NUM_CALIBRATION_SAMPLES * 0.5)) +
    pick(medium, int(NUM_CALIBRATION_SAMPLES * 0.3)) +
    pick(long_,  int(NUM_CALIBRATION_SAMPLES * 0.2))
)
while len(chosen) < NUM_CALIBRATION_SAMPLES:
    leftover = [r for r in pool if r not in chosen]
    if not leftover:
        break
    chosen.append(rng.choice(leftover))
rng.shuffle(chosen)
chosen = chosen[:NUM_CALIBRATION_SAMPLES]
print(f"Selected {len(chosen)} calibration samples")


# ============== compute angular distances ==============
# PruneMe approach: run forward pass with output_hidden_states=True,
# then extract last non-padded token per layer, compute block distances.
#
# Unlike PruneMe which batches generic text, we process one sample at a time
# because G-code samples have very different lengths (500-50000 tokens).
# This avoids excessive padding waste.

print(f"\n=== Computing angular distances (block size = {LAYERS_TO_SKIP}) ===")

# Accumulate distances across all samples
# all_distances[i] = list of distances for starting position i
all_distances = [[] for _ in range(num_layers - LAYERS_TO_SKIP)]

with torch.no_grad():
    for idx, sample in enumerate(tqdm(chosen, desc="Calibration")):
        ids = sample["input_ids"]
        if len(ids) > MAX_SEQ_LENGTH:
            ids = ids[:MAX_SEQ_LENGTH]
        
        input_ids = torch.tensor(ids, dtype=torch.long, device="cuda:0").unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)  # no padding since single sample
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states   # tuple of (L+1) tensors, each (1, T, H)
        
        # Extract last non-padded token per layer
        # (faithful to PruneMe's get_last_non_padded_tokens)
        last_tokens = get_last_non_padded_tokens(hidden_states, attention_mask)
        
        # Remove embedding layer (index 0) — PruneMe does this too (line 65):
        # "Remove the first element to account for the input layer"
        last_tokens = last_tokens[1:]
        assert len(last_tokens) == num_layers, \
            f"Expected {num_layers} layers, got {len(last_tokens)}"
        
        # Compute block distances for this sample
        # (faithful to PruneMe's compute_block_distances)
        distances = compute_block_distances(last_tokens, LAYERS_TO_SKIP)
        for i, d in enumerate(distances):
            all_distances[i].append(d)

# Average over all calibration samples (PruneMe line 77)
average_distances = [np.mean(dists) for dists in all_distances]


# ============== find best block ==============
print(f"\n=== Block Angular Distances (block size = {LAYERS_TO_SKIP}) ===")

min_distance = float('inf')
min_start = 0

for i, avg_dist in enumerate(average_distances):
    # PruneMe uses 1-based indexing in output; we use 0-based internally
    marker = ""
    if avg_dist < min_distance:
        min_distance = avg_dist
        min_start = i
    print(f"  Block [{i:2d}, {i + LAYERS_TO_SKIP:2d}): avg_distance = {avg_dist:.6f}")

to_remove = list(range(min_start, min_start + LAYERS_TO_SKIP))
to_keep = [i for i in range(num_layers) if i not in to_remove]

print(f"\n★ Best block to remove: layers {to_remove}")
print(f"  Angular distance: {min_distance:.6f}")
print(f"  Layers kept: {to_keep}")


# ============== prune model ==============
print(f"\nPruning: {num_layers} → {num_layers - LAYERS_TO_SKIP} layers")

# Disable output_hidden_states for the pruned model (not needed for inference)
model.config.output_hidden_states = False

new_layers = torch.nn.ModuleList([
    model.model.layers[i] for i in range(num_layers) if i not in to_remove
])
model.model.layers = new_layers
model.config.num_hidden_layers = len(new_layers)


# ============== untie lm_head & save ==============
SAVE_DIR = f"granite3b-pruneme-skip{LAYERS_TO_SKIP}-block{min_start}to{min_start + LAYERS_TO_SKIP - 1}"

# Untie lm_head (Granite uses tied embeddings)
if model.config.tie_word_embeddings:
    print("Untying lm_head...")
    embed_w = model.model.embed_tokens.weight.data.clone()
    model.lm_head.weight = torch.nn.Parameter(embed_w)
    model.config.tie_word_embeddings = False

Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
print(f"Saving to {SAVE_DIR}/")
model.save_pretrained(SAVE_DIR, safe_serialization=True)
tokenizer.save_pretrained(SAVE_DIR)

# Save analysis log
with open(f"{SAVE_DIR}/pruneme_log.json", "w") as f:
    json.dump({
        "model_id": MODEL_ID,
        "algorithm": "Gromov et al. ICLR 2025 / PruneMe (arcee-ai)",
        "num_layers_original": num_layers,
        "layers_to_skip": LAYERS_TO_SKIP,
        "best_block_start": min_start,
        "best_block_end": min_start + LAYERS_TO_SKIP - 1,
        "layers_removed": to_remove,
        "layers_kept": to_keep,
        "min_angular_distance": min_distance,
        "all_block_distances": [
            {
                "block_start": i,
                "block_end": i + LAYERS_TO_SKIP - 1,
                "avg_distance": average_distances[i],
            }
            for i in range(len(average_distances))
        ],
        "calibration": {
            "num_samples": len(chosen),
            "max_seq_length": MAX_SEQ_LENGTH,
            "data_path": DATA_PATH,
        },
    }, f, indent=2)

# Final size
import os
total_bytes = sum(
    os.path.getsize(os.path.join(SAVE_DIR, f))
    for f in os.listdir(SAVE_DIR)
    if os.path.isfile(os.path.join(SAVE_DIR, f))
)
reduction = LAYERS_TO_SKIP / num_layers * 100
print(f"\nFinal size: {total_bytes / 1e9:.2f} GB (removed {reduction:.1f}% layers)")
print(f"Log: {SAVE_DIR}/pruneme_log.json")
print("Done.")
