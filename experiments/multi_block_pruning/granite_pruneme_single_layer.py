"""Rank individual Granite decoder layers by input/output angular distance.

Score the last valid token of each calibration sample in the original model,
then remove the K lowest-scoring layers simultaneously. Indices are zero-based.
"""

import json
import math
import random
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "/home/aichen/project_data/model/granite_gcode_merged_best_new"
DATA_PATH = "/home/aichen/project_data/training_data/dataset_no_rule/train.jsonl"
NUM_CALIBRATION_SAMPLES = 128
MAX_SEQ_LENGTH = 65536
TOTAL_PRUNE_LAYERS = 16
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


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
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
    rank_layers([0.0] * num_layers, TOTAL_PRUNE_LAYERS)
    print(f"Plan: independently rank {num_layers} layers; remove {TOTAL_PRUNE_LAYERS}")

    samples = build_calibration_data(tokenizer)
    scores = compute_layer_scores(model, samples)
    ranking, removed, kept = rank_layers(scores, TOTAL_PRUNE_LAYERS)
    layers_by_score = {}
    for index in ranking:
        layers_by_score.setdefault(scores[index], []).append(index)
    tied_groups = [
        (score, indices) for score, indices in layers_by_score.items()
        if len(indices) > 1
    ]
    print("\nExact score ties:")
    if tied_groups:
        for score, indices in tied_groups:
            print(f"  distance={score!r}, layers={indices}")
        print("  Ties are resolved by preferring the higher original layer index.")
    else:
        print("  No exactly equal layer scores.")
        
    print("\nLayer ranking (least important first; original zero-based indices):")
    for rank, index in enumerate(ranking, start=1):
        status = "REMOVE" if index in removed else "KEEP"
        print(f"  {rank:2d}. Layer {index:2d}: distance={scores[index]:.6f} {status}")
    print(f"Layers removed: {removed}")
    print(f"Layers kept: {kept}")

    model.model.layers = torch.nn.ModuleList([model.model.layers[i] for i in kept])
    model.config.num_hidden_layers = len(kept)
    model.model.config.num_hidden_layers = len(kept)
    model.config.output_hidden_states = False
    # Cache indices must refer to positions in the newly pruned model.
    for new_index, layer in enumerate(model.model.layers):
        if hasattr(layer, "layer_idx"):
            layer.layer_idx = new_index
        if hasattr(layer.self_attn, "layer_idx"):
            layer.self_attn.layer_idx = new_index

    if model.config.tie_word_embeddings:
        model.lm_head.weight = torch.nn.Parameter(
            model.model.embed_tokens.weight.detach().clone(),
        )
        model.config.tie_word_embeddings = False

    save_dir = Path(__file__).resolve().parent / "outputs" / (
        f"granite3b-pruneme-single-layer-skip{TOTAL_PRUNE_LAYERS}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir, safe_serialization=True)
    tokenizer.save_pretrained(save_dir)
    log = {
        "model_id": MODEL_ID,
        "algorithm": "Per-layer input/output angular distance, lowest-K selection",
        "pruning_type": "single_layer",
        "selection_method": "global_ranking_on_original_model",
        "index_base": 0,
        "num_layers_original": num_layers,
        "total_prune_layers": TOTAL_PRUNE_LAYERS,
        "layers_removed": removed,
        "layers_kept": kept,
        "new_to_original_layer": kept,
        "layer_ranking": [
            {"rank": rank, "layer": index, "avg_distance": scores[index],
             "removed": index in removed}
            for rank, index in enumerate(ranking, start=1)
        ],
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
    print(f"Layers: {num_layers} -> {len(kept)}, size: {total_bytes / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
