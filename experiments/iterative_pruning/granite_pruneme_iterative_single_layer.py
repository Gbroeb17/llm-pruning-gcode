"""Iterative Granite single-layer pruning using the original layer-wise scoring.

Each round re-scores all remaining layers on the same calibration samples,
prunes the lowest-scoring PRUNE_PER_ROUND layers, and tracks original indices.
"""

import json
import math
import random
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "/media/ssd2/llm-pruning/shared/models/granite_gcode_merged_best_new"
DATA_PATH = "/media/ssd2/llm-pruning/shared/datasets/training_data/dataset_no_rule/train.jsonl"
NUM_CALIBRATION_SAMPLES = 128
MAX_SEQ_LENGTH = 65536
PRUNE_PER_ROUND = 3  # x: layers removed per iteration
TOTAL_PRUNE_LAYERS = 12  # y: total layers removed
MIN_LEN = 1024
SEED = 42


def angular_distance(layer_input, layer_output):
    x = layer_input / torch.norm(layer_input, dim=-1, keepdim=True)
    y = layer_output / torch.norm(layer_output, dim=-1, keepdim=True)
    cosine_similarity = (x * y).sum(-1)
    return torch.acos(cosine_similarity.clamp(min=-1, max=1)) / torch.pi


def build_calibration_data(tokenizer):
    if NUM_CALIBRATION_SAMPLES <= 0 or MAX_SEQ_LENGTH <= 0:
        raise ValueError("Sample count and maximum sequence length must be positive.")
    raw = []
    with open(DATA_PATH, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            text = tokenizer.apply_chat_template(
                row["messages"], tokenize=False, add_generation_prompt=False,
            )
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            raw.append({"input_ids": ids, "tok_len": len(ids)})

    pool = [i for i, row in enumerate(raw) if row["tok_len"] >= MIN_LEN]
    short = [i for i in pool if raw[i]["tok_len"] < 3600]
    medium = [i for i in pool if 3600 <= raw[i]["tok_len"] < 20000]
    long = [i for i in pool if raw[i]["tok_len"] >= 20000]
    print(f"Pool: {len(pool)}, buckets: {len(short)}/{len(medium)}/{len(long)}")
    rng = random.Random(SEED)
    chosen = (
        rng.sample(short, min(int(NUM_CALIBRATION_SAMPLES * 0.5), len(short)))
        + rng.sample(medium, min(int(NUM_CALIBRATION_SAMPLES * 0.3), len(medium)))
        + rng.sample(long, min(int(NUM_CALIBRATION_SAMPLES * 0.2), len(long)))
    )
    selected = set(chosen)
    leftover = [i for i in pool if i not in selected]
    remaining = min(NUM_CALIBRATION_SAMPLES - len(chosen), len(leftover))
    chosen.extend(rng.sample(leftover, remaining))
    rng.shuffle(chosen)
    if not chosen:
        raise ValueError("No calibration samples meet MIN_LEN; check the dataset.")
    print(f"Selected {len(chosen)} calibration samples")
    return [raw[i] for i in chosen]


def compute_layer_scores(model, samples):
    layers = model.model.layers
    totals = [0.0] * len(layers)
    counts = [0] * len(layers)
    captured_inputs = {}
    handles = []
    device = model.model.embed_tokens.weight.device

    def make_input_hook(index):
        def save_input(module, args, kwargs):
            hidden = args[0] if args else kwargs["hidden_states"]
            # Clone only the last token, without retaining the full sequence.
            captured_inputs[index] = hidden[:, -1, :].detach().clone()
        return save_input

    def make_output_hook(index):
        def score_output(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            distance = angular_distance(
                captured_inputs.pop(index), hidden[:, -1, :],
            ).mean().item()
            if not math.isfinite(distance):
                raise ValueError(f"Non-finite angular distance at layer {index}.")
            totals[index] += distance
            counts[index] += 1
        return score_output

    try:
        for index, layer in enumerate(layers):
            handles.append(layer.register_forward_pre_hook(
                make_input_hook(index), with_kwargs=True,
            ))
            handles.append(layer.register_forward_hook(make_output_hook(index)))

        with torch.inference_mode():
            for sample in tqdm(samples, desc="Single-layer calibration"):
                ids = sample["input_ids"][:MAX_SEQ_LENGTH]
                input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
                # Each forward uses one unpadded sample, so -1 is the last valid token.
                outputs = model.model(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    use_cache=False,
                    output_hidden_states=False,
                    output_attentions=False,
                    return_dict=True,
                )
                del outputs
    finally:
        for handle in handles:
            handle.remove()
        captured_inputs.clear()

    if any(count != len(samples) for count in counts):
        raise RuntimeError("Not every decoder layer was scored for every sample.")
    return [total / count for total, count in zip(totals, counts)]

#分數完全相同時，優先剪後面的層（層號較大）
def rank_layers(scores, prune_count):
    if not 0 <= prune_count < len(scores):
        raise ValueError("TOTAL_PRUNE_LAYERS must be >= 0 and less than the layer count.")
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("All layer scores must be finite.")
    ranking = sorted(range(len(scores)), key=lambda i: (scores[i], -i))
    removed = sorted(ranking[:prune_count])
    removed_set = set(removed)
    kept = [i for i in range(len(scores)) if i not in removed_set]
    return ranking, removed, kept


def update_layers(model, kept_indices):
    """Remove layers and refresh indices used by KV cache and attention."""
    model.model.layers = torch.nn.ModuleList(
        [model.model.layers[i] for i in kept_indices]
    )
    count = len(kept_indices)
    model.config.num_hidden_layers = count
    model.model.config.num_hidden_layers = count
    model.config.output_hidden_states = False
    for new_index, layer in enumerate(model.model.layers):
        if hasattr(layer, "layer_idx"):
            layer.layer_idx = new_index
        if hasattr(layer.self_attn, "layer_idx"):
            layer.self_attn.layer_idx = new_index


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    if PRUNE_PER_ROUND <= 0:
        raise ValueError("PRUNE_PER_ROUND must be positive.")

    print(f"Loading model from {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_layers = len(model.model.layers)
    if not 0 < TOTAL_PRUNE_LAYERS < num_layers:
        raise ValueError("TOTAL_PRUNE_LAYERS must be between 1 and num_layers - 1.")
    print(f"Plan: single-layer iterative pruning 1:{PRUNE_PER_ROUND}:{TOTAL_PRUNE_LAYERS}")

    # Important for fair comparison: draw calibration data exactly once.
    samples = build_calibration_data(tokenizer)
    original_indices = list(range(num_layers))
    removed_original = []
    rounds = []

    while len(removed_original) < TOTAL_PRUNE_LAYERS:
        round_number = len(rounds) + 1
        current_count = len(original_indices)
        count_to_remove = min(
            PRUNE_PER_ROUND, TOTAL_PRUNE_LAYERS - len(removed_original)
        )
        print(f"\n=== Round {round_number}: {current_count} layers, "
              f"remove {count_to_remove} ===", flush=True)

        # Unmodified original angular_distance + hook-based scoring.
        scores = compute_layer_scores(model, samples)
        ranking, removed_current, kept_current = rank_layers(scores, count_to_remove)
        removed_this_round = [original_indices[i] for i in removed_current]
        kept_original = [original_indices[i] for i in kept_current]

        print("Ranking (current index -> original index, angular distance):")
        removed_set = set(removed_current)
        for rank, current_index in enumerate(ranking, 1):
            print(f" {rank:2d}. {current_index:2d} -> "
                  f"L{original_indices[current_index]:02d}: "
                  f"{scores[current_index]:.6f} "
                  f"{'REMOVE' if current_index in removed_set else 'KEEP'}")
        print(f"Removed original layers this round: {removed_this_round}", flush=True)

        rounds.append({
            "round": round_number,
            "num_layers_before": current_count,
            "removed_current_indices": removed_current,
            "removed_original_indices": removed_this_round,
            "kept_original_indices": kept_original,
            "ranking": [
                {"rank": rank, "current_layer": i,
                 "original_layer": original_indices[i],
                 "avg_distance": scores[i], "removed": i in removed_set}
                for rank, i in enumerate(ranking, 1)
            ],
            "exact_score_ties": [
                {"score": score, "original_layers": [
                    original_indices[i] for i, value in enumerate(scores)
                    if value == score
                ]}
                for score in sorted(set(scores)) if scores.count(score) > 1
            ],
        })
        removed_original.extend(removed_this_round)
        update_layers(model, kept_current)
        original_indices = kept_original

    if model.config.tie_word_embeddings:
        model.lm_head.weight = torch.nn.Parameter(
            model.model.embed_tokens.weight.detach().clone(),
        )
        model.config.tie_word_embeddings = False

    save_dir = Path(__file__).resolve().parent / "outputs" / (
        f"granite3b-pruneme-iterative-1x{PRUNE_PER_ROUND}x{TOTAL_PRUNE_LAYERS}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir, safe_serialization=True)
    tokenizer.save_pretrained(save_dir)
    log = {
        "model_id": MODEL_ID,
        "algorithm": "Iterative per-layer input/output angular distance",
        "pruning_type": "iterative_single_layer",
        "selection_method": "rescore_after_each_round",
        "index_base": 0,
        "num_layers_original": num_layers,
        "prune_per_round": PRUNE_PER_ROUND,
        "total_prune_layers": TOTAL_PRUNE_LAYERS,
        "layers_removed": sorted(removed_original),
        "layers_removed_in_order": removed_original,
        "layers_kept": original_indices,
        "new_to_original_layer": original_indices,
        "rounds": rounds,
        "calibration": {
            "num_samples": len(samples),
            "max_seq_length": MAX_SEQ_LENGTH,
            "data_path": DATA_PATH,
            "seed": SEED,
            "token_selection": "last_non_padded_token",
            "truncation": "prefix",
        },
    }
    with (save_dir / "pruneme_log.json").open("w", encoding="utf-8") as handle:
        json.dump(log, handle, indent=2)
    total_bytes = sum(path.stat().st_size for path in save_dir.iterdir() if path.is_file())
    print(f"Saved to: {save_dir}")
    print(f"Original layers removed: {sorted(removed_original)}")
    print(f"Original layers kept: {original_indices}")
    print(f"Layers: {num_layers} -> {len(original_indices)}, size: "
          f"{total_bytes / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
