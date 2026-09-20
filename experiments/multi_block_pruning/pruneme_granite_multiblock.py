"""
Multi-block layer pruning for Granite-Code-3B on the G-code task.

Similarity scoring is based on:
  - Paper: "The Unreasonable Ineffectiveness of the Deeper Layers"
           (Gromov et al., ICLR 2025)
  - Code: arcee-ai/PruneMe (unofficial implementation)

Angular-distance scoring follows the original PruneMe idea, while this
script extends the block-selection stage from one contiguous block to
multiple non-overlapping blocks.

Algorithm:

  1. Load the full Granite model with output_hidden_states=True.

  2. For each calibration sample, extract the final non-padded token's
     hidden representation at every decoder layer.

  3. For every possible sliding N-layer block starting at layer l,
     compute:

       d(l, l+N) = (1/pi) * arccos(cos_sim(h[l], h[l+N]))

     and average the distance over all calibration samples.

  4. Rank all candidate blocks by angular distance.
     Smaller distance means that the hidden representation changes less
     across the block and therefore suggests higher redundancy.

  5. Greedily select multiple blocks in ascending distance order.
     Selected blocks may be adjacent, but they may not overlap.

  6. Continue until TOTAL_PRUNE_LAYERS layers have been selected.

  7. Remove all selected layers from model.model.layers and save the
     resulting multi-block-pruned model.

For the current 2+2+2+2+2+2 experiment:
  - BLOCK_SIZE = 2
  - TOTAL_PRUNE_LAYERS = 12
  - NUM_BLOCKS = 6
  - Granite: 32 -> 20 layers (37.5% layer pruning)

Streamline recovery is performed separately after pruning.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============== config ==============
MODEL_ID = "/home/aichen/project_data/model/granite_gcode_merged_best_new"
DATA_PATH = "/home/aichen/project_data/training_data/dataset_no_rule/train.jsonl"

NUM_CALIBRATION_SAMPLES = 128
MAX_SEQ_LENGTH = 65536

# Multi-block pruning configuration:
#   BLOCK_SIZE = number of consecutive layers in each candidate block
#   TOTAL_PRUNE_LAYERS = total number of layers to remove
#   NUM_BLOCKS = number of non-overlapping blocks to select
#
# For 2+2+2+2+2+2:
#   BLOCK_SIZE = 2
#   TOTAL_PRUNE_LAYERS = 12
#   NUM_BLOCKS = 6
BLOCK_SIZE = 4
TOTAL_PRUNE_LAYERS = 12
NUM_BLOCKS = TOTAL_PRUNE_LAYERS // BLOCK_SIZE

# Each selected block must have the same BLOCK_SIZE,
# so the total pruning amount must be divisible by BLOCK_SIZE.
if TOTAL_PRUNE_LAYERS % BLOCK_SIZE != 0:
    raise ValueError(
        "TOTAL_PRUNE_LAYERS must be divisible by BLOCK_SIZE"
    )

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


def compute_block_distances(hidden_states, block_size: int):
    """Compute angular distance for every possible sliding block.

    For a block starting at layer l with size N, compare:
        h[l] vs. h[l + N]

    A smaller angular distance means the representation changes less
    after passing through the N-layer block, suggesting that the block
    may contain more redundant computation.

    Example for BLOCK_SIZE = 2:
        candidate [0, 1] -> compare h[0] and h[2]
        candidate [1, 2] -> compare h[1] and h[3]
        candidate [2, 3] -> compare h[2] and h[4]
        ...

    Args:
        hidden_states: list of (B, H) tensors, one per decoder layer
        block_size: number of consecutive layers in each candidate block

    Returns:
        list of float, one angular distance for each possible block start
    """
    distances = []
    num_layers = len(hidden_states)

    for l in range(num_layers - block_size):
        block_distance = angular_distance(
            hidden_states[l],
            hidden_states[l + block_size]
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
print(
    f"Plan: select {NUM_BLOCKS} non-overlapping blocks, "
    f"{BLOCK_SIZE} layers per block, "
    f"{TOTAL_PRUNE_LAYERS} layers removed in total"
)


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

print(f"\n=== Computing angular distances (block size = {BLOCK_SIZE}) ===")

# Accumulate distances across all samples
# all_distances[i] = list of distances for starting position i
all_distances = [[] for _ in range(num_layers - BLOCK_SIZE)]

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
        distances = compute_block_distances(last_tokens, BLOCK_SIZE)
        for i, d in enumerate(distances):
            all_distances[i].append(d)

# Average over all calibration samples (PruneMe line 77)
average_distances = [np.mean(dists) for dists in all_distances]


# ============== select multiple pruning blocks ==============
print(f"\n=== Block Angular Distances (block size = {BLOCK_SIZE}) ===")

# Build one candidate for every possible sliding block.
#
# Example for BLOCK_SIZE = 2:
#   [0,1], [1,2], [2,3], ..., [29,30]
#
# Each candidate stores:
#   - start: first layer of the block
#   - end:   last layer of the block
#   - distance: average angular distance across calibration samples
candidates = []

for start, avg_dist in enumerate(average_distances):
    end = start + BLOCK_SIZE - 1

    candidate = {
        "start": start,
        "end": end,
        "distance": float(avg_dist),
    }

    candidates.append(candidate)

    print(
        f"  Block [{start:2d}, {end:2d}]: "
        f"avg_distance = {avg_dist:.6f}"
    )


# Smaller angular distance means the representation before and after
# the block is more similar.
#
# Therefore, blocks with smaller distances receive higher pruning priority.
# If two blocks have exactly the same distance, prefer the earlier layer.
candidates.sort(key=lambda x: (x["distance"], x["start"]))


# Greedily select the best blocks while preventing overlap.
#
# Important:
#   Overlap is NOT allowed:
#       [4,5] + [5,6] -> invalid
#
#   Direct adjacency IS allowed:
#       [4,5] + [6,7] -> valid
#
# Adjacency is intentionally allowed because block selection is performed
# independently of the later Streamline recovery stage.
selected_blocks = []
used_layers = set()

for candidate in candidates:
    start = candidate["start"]
    end = candidate["end"]

    block_layers = set(range(start, end + 1))

    # Skip this candidate if any of its layers has already been selected
    # by a higher-priority block.
    if block_layers & used_layers:
        continue

    selected_blocks.append(candidate)
    used_layers.update(block_layers)

    # Stop after selecting enough blocks to reach the target pruning amount.
    if len(selected_blocks) == NUM_BLOCKS:
        break


# Make sure the greedy selection successfully found enough
# non-overlapping blocks.
if len(selected_blocks) != NUM_BLOCKS:
    raise RuntimeError(
        f"Could only select {len(selected_blocks)} valid blocks, "
        f"but {NUM_BLOCKS} blocks are required."
    )


# Sort by original layer position only for easier reading and logging.
# This does NOT change which blocks were selected.
selected_blocks.sort(key=lambda x: x["start"])


# Flatten the selected blocks into one list of original layer indices.
#
# Example:
#   selected_blocks = [2-3, 8-9, 14-15]
#   to_remove       = [2, 3, 8, 9, 14, 15]
to_remove = sorted(
    layer
    for block in selected_blocks
    for layer in range(block["start"], block["end"] + 1)
)

to_keep = [
    i for i in range(num_layers)
    if i not in to_remove
]


# Safety checks:
#   1. Exactly TOTAL_PRUNE_LAYERS must be removed.
#   2. Every removed layer must be unique.
assert len(to_remove) == TOTAL_PRUNE_LAYERS, (
    f"Expected to prune {TOTAL_PRUNE_LAYERS} layers, "
    f"but got {len(to_remove)}."
)

assert len(set(to_remove)) == TOTAL_PRUNE_LAYERS, (
    "Duplicate layers detected in pruning plan."
)


print("\n★ Selected multi-block pruning plan")

for rank, block in enumerate(selected_blocks, start=1):
    print(
        f"  Block {rank}: "
        f"layers [{block['start']}, {block['end']}], "
        f"angular_distance = {block['distance']:.6f}"
    )

print(f"\nLayers removed ({len(to_remove)}): {to_remove}")
print(f"Layers kept ({len(to_keep)}): {to_keep}")

# Identify adjacent selected blocks for later Streamline processing.
# This does not change the pruning result.
# It only records whether two selected blocks are directly adjacent.
adjacent_pairs = []

for i in range(len(selected_blocks) - 1):
    current_block = selected_blocks[i]
    next_block = selected_blocks[i + 1]

    if current_block["end"] + 1 == next_block["start"]:
        adjacent_pairs.append({
            "left_block": [
                current_block["start"],
                current_block["end"],
            ],
            "right_block": [
                next_block["start"],
                next_block["end"],
            ],
        })

if adjacent_pairs:
    print("\nAdjacent selected blocks:")
    for pair in adjacent_pairs:
        print(
            f"  {pair['left_block']} + "
            f"{pair['right_block']}"
        )
else:
    print("\nNo adjacent selected blocks.")

# ============== prune model ==============
print(
    f"\nPruning: {num_layers} → "
    f"{num_layers - TOTAL_PRUNE_LAYERS} layers"
)

# Disable output_hidden_states for the pruned model (not needed for inference)
model.config.output_hidden_states = False

new_layers = torch.nn.ModuleList([
    model.model.layers[i] for i in range(num_layers) if i not in to_remove
])
model.model.layers = new_layers
model.config.num_hidden_layers = len(new_layers)


# ============== untie lm_head & save ==============
OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"

SAVE_DIR = OUTPUT_ROOT / (
    f"granite3b-pruneme-multiblock-"
    f"b{BLOCK_SIZE}-"
    f"n{NUM_BLOCKS}-"
    f"skip{TOTAL_PRUNE_LAYERS}"
)

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

# Save pruning analysis log.
#
# This file is important for the later Streamline stage because it records
# exactly which original blocks and layers were removed.
with open(f"{SAVE_DIR}/pruneme_log.json", "w") as f:
    json.dump({
        "model_id": MODEL_ID,

        # PruneMe angular distance is still used for block scoring,
        # but the original single-block selection is replaced by
        # greedy multi-block selection.
        "algorithm": (
            "PruneMe angular-distance scoring + "
            "multi-block greedy non-overlap selection"
        ),

        "pruning_type": "multi_block",
        "selection_method": "greedy_nonoverlap",

        "num_layers_original": num_layers,

        # Multi-block configuration
        "block_size": BLOCK_SIZE,
        "total_prune_layers": TOTAL_PRUNE_LAYERS,
        "num_blocks": NUM_BLOCKS,

        # Final selected pruning plan
        "selected_blocks": selected_blocks,
        "adjacent_pairs": adjacent_pairs,
        "layers_removed": to_remove,
        "layers_kept": to_keep,

        # Keep all candidate distances for later analysis.
        # This allows us to inspect why particular blocks were selected.
        "all_block_distances": [
            {
                "block_start": i,
                "block_end": i + BLOCK_SIZE - 1,
                "avg_distance": float(average_distances[i]),
            }
            for i in range(len(average_distances))
        ],

        # Calibration setup used to compute representation similarity.
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
reduction = TOTAL_PRUNE_LAYERS / num_layers * 100
print(f"\nFinal size: {total_bytes / 1e9:.2f} GB (removed {reduction:.1f}% layers)")
print(f"Log: {SAVE_DIR}/pruneme_log.json")
print("Done.")
