"""Iterative Granite pruning: score all 3-layer blocks, remove one, repeat.

Requirements: torch, transformers, accelerate, tqdm, numpy; Python >= 3.10.
Example:
  python iterative_multiblock_pruning.py --model-id /path/model \
      --data-path /path/train.jsonl --output-dir /path/pruned

Compatibility version: intentionally preserves the uploaded multiblock's
hidden-state indexing, native scoring dtype, L-N candidates, NumPy averaging,
and calibration sampling (including token-content duplicate exclusion).
After removing embedding state, compare last_tokens[l] and last_tokens[l+3],
but remove current decoder layers [l, l+1, l+2], exactly as the source does.
The final returned hidden state includes the decoder final normalization.
This legacy scoring/index convention is retained for controlled comparison.
Calibration is drawn once and reused; only block selection becomes iterative.
Full hidden states/logits and the model's default cache behavior are retained,
so memory use can be high at 65536 tokens, as in the source implementation.
"""

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ================= User settings =================
MODEL_ID = "/media/ssd2/llm-pruning/shared/models/granite_gcode_merged_best_new"
DATA_PATH = "/media/ssd2/llm-pruning/shared/datasets/training_data/dataset_no_rule/train.jsonl"
BLOCK_SIZE = 3
TOTAL_PRUNE_LAYERS = 12


def default_output_dir():
    return Path(__file__).resolve().parent / "outputs" / (
        f"granite3b-pruneme-iterative-multiblock-b{BLOCK_SIZE}-"
        f"n{TOTAL_PRUNE_LAYERS // BLOCK_SIZE}-skip{TOTAL_PRUNE_LAYERS}"
    )


def angular_distance(x, y):
    x_norm = x / torch.norm(x, dim=-1, keepdim=True)
    y_norm = y / torch.norm(y, dim=-1, keepdim=True)
    cosine_similarity = (x_norm * y_norm).sum(-1)
    return torch.acos(cosine_similarity.clamp(min=-1, max=1)) / torch.pi


def build_calibration_data(tokenizer, args):
    raw = []
    source_lines = {}
    with Path(args.data_path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                text = tokenizer.apply_chat_template(
                    row["messages"], tokenize=False, add_generation_prompt=False
                )
                ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            except Exception as exc:
                raise ValueError(f"Invalid calibration row at line {line_number}") from exc
            if len(ids) >= args.min_len:
                sample = {"input_ids": ids, "tok_len": len(ids)}
                raw.append(sample)
                source_lines[id(sample)] = line_number
    buckets = [
        [s for s in raw if s["tok_len"] < 3600],
        [s for s in raw if 3600 <= s["tok_len"] < 20000],
        [s for s in raw if s["tok_len"] >= 20000],
    ]
    rng = random.Random(args.seed)
    chosen = []
    for bucket, fraction in zip(buckets, (0.5, 0.3, 0.2)):
        chosen.extend(rng.sample(bucket, min(int(args.num_samples * fraction), len(bucket))))
    # Match original dictionary equality, RNG calls, and one-at-a-time refill.
    # Source-line metadata is deliberately kept outside these dictionaries.
    while len(chosen) < args.num_samples:
        leftover = [s for s in raw if s not in chosen]
        if not leftover:
            break
        chosen.append(rng.choice(leftover))
    rng.shuffle(chosen)
    samples = chosen[:args.num_samples]
    if not samples:
        raise ValueError("No calibration samples meet min_len.")
    metadata = {
        "data_path": str(Path(args.data_path).resolve()), "seed": args.seed,
        "requested_samples": args.num_samples, "num_samples": len(samples),
        "eligible_samples": len(raw), "bucket_sizes": list(map(len, buckets)),
        "bucket_boundaries": [3600, 20000], "target_fractions": [0.5, 0.3, 0.2],
        "min_len": args.min_len, "max_seq_length": args.max_seq_length,
        "truncation": "prefix", "token_selection": "last_non_padded_token",
        "samples": [{"source_line": source_lines[id(s)], "original_tokens": s["tok_len"],
                     "used_tokens": min(s["tok_len"], args.max_seq_length),
                     "used_token_ids_sha256": hashlib.sha256(
                         json.dumps(s["input_ids"][:args.max_seq_length],
                                    separators=(",", ":")).encode()).hexdigest()}
                    for s in samples],
    }
    print(f"Calibration: {len(samples)} samples; buckets {list(map(len, buckets))}")
    return samples, metadata


def compute_block_scores(model, samples, max_seq_length):
    # Keep original multiblock scoring literally, including its index offset
    # and L-N (rather than L-N+1) candidate range.
    num_layers = len(model.model.layers)
    all_distances = [[] for _ in range(num_layers - BLOCK_SIZE)]
    device = model.model.embed_tokens.weight.device
    with torch.no_grad():
        for sample in tqdm(samples, desc="Block calibration"):
            ids = torch.tensor(sample["input_ids"][:max_seq_length],
                               dtype=torch.long, device=device).unsqueeze(0)
            attention_mask = torch.ones_like(ids)
            outputs = model(input_ids=ids, attention_mask=attention_mask,
                            output_hidden_states=True, return_dict=True)
            if len(outputs.hidden_states) != num_layers + 1:
                raise RuntimeError("Expected L+1 hidden states from Granite decoder.")
            last_tokens = []
            for layer_hidden in outputs.hidden_states:
                batch_last_tokens = []
                for batch_index in range(layer_hidden.size(0)):
                    last_index = attention_mask[batch_index].nonzero(as_tuple=True)[0].max()
                    batch_last_tokens.append(
                        layer_hidden[batch_index, last_index, :].unsqueeze(0))
                last_tokens.append(torch.cat(batch_last_tokens, dim=0))
            last_tokens = last_tokens[1:]
            for start in range(num_layers - BLOCK_SIZE):
                distance = angular_distance(
                    last_tokens[start], last_tokens[start + BLOCK_SIZE]).mean().item()
                if not math.isfinite(distance):
                    raise ValueError(f"Non-finite score at block {start}.")
                all_distances[start].append(distance)
            del outputs, last_tokens, batch_last_tokens, layer_hidden, ids, attention_mask
    if not all_distances or any(len(values) != len(samples) for values in all_distances):
        raise RuntimeError("Every candidate must be scored on every sample.")
    return [float(np.mean(values)) for values in all_distances]


def rebuild_layers(model, kept):
    """Rebuild ModuleList in place, retaining trained weights without reload."""
    model.model.layers = torch.nn.ModuleList([model.model.layers[i] for i in kept])
    for config in (model.config, model.model.config):
        config.num_hidden_layers = len(kept)
        config.output_hidden_states = False
    for index, layer in enumerate(model.model.layers):
        if hasattr(layer, "layer_idx"):
            layer.layer_idx = index
        # Guard absent attention modules and update nested attention indices too.
        for module in layer.modules():
            if hasattr(module, "layer_idx"):
                module.layer_idx = index
    if len(model.model.layers) != model.config.num_hidden_layers:
        raise RuntimeError("Decoder/config layer count mismatch.")


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False),
                         encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--data-path", default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Optional override; otherwise auto-named from pruning settings")
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--max-seq-length", type=int, default=65536)
    parser.add_argument("--min-len", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0", help="Single device, e.g. cuda:0 or cpu")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    args = parser.parse_args()
    if BLOCK_SIZE <= 0 or TOTAL_PRUNE_LAYERS <= 0 or TOTAL_PRUNE_LAYERS % BLOCK_SIZE:
        parser.error("Positive TOTAL_PRUNE_LAYERS must be divisible by positive BLOCK_SIZE.")
    if args.output_dir is None:
        args.output_dir = default_output_dir()
    if min(args.num_samples, args.max_seq_length, args.min_len) <= 0:
        parser.error("Calibration sample count and lengths must be positive.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory must be empty to avoid mixing checkpoints/logs.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, dtype=getattr(torch, args.dtype),
        device_map=args.device, low_cpu_mem_usage=True, output_hidden_states=True)
    model.eval()
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("Expected Granite-style model.model.layers decoder.")
    num_layers = len(model.model.layers)
    if not 0 < TOTAL_PRUNE_LAYERS < num_layers or TOTAL_PRUNE_LAYERS % BLOCK_SIZE:
        raise ValueError("Pruning must remove full blocks and leave at least one layer.")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    samples, calibration = build_calibration_data(tokenizer, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mapping = list(range(num_layers))
    removed = []
    log = {
        "model_id": args.model_id, "pruning_type": "iterative_multi_block",
        "algorithm": "Original multiblock hidden-state angular distance",
        "scoring_compatibility": "uploaded_multiblock_legacy",
        "scoring_endpoints": "outputs.hidden_states[1:][l] versus [l+block_size]",
        "candidate_count": "current_num_layers - block_size",
        "score_precision": "native model hidden-state dtype; no FP32 cast",
        "sample_aggregation": "numpy.mean of per-sample distances",
        "cache_behavior": "model default, matching source",
        "numpy_version": np.__version__,
        "selection_method": "remove_one_best_block_then_rebuild_and_rescore",
        "consecutiveness": "current_decoder_indices", "index_base": 0,
        "tie_break": "earlier_current_block_start", "block_size": BLOCK_SIZE,
        "total_prune_layers": TOTAL_PRUNE_LAYERS, "num_layers_original": num_layers,
        "dtype": args.dtype, "device": args.device,
        "torch_version": str(torch.__version__), "calibration": calibration,
        "rounds": [], "status": "pruning",
    }
    for round_number in range(1, TOTAL_PRUNE_LAYERS // BLOCK_SIZE + 1):
        print(f"Round {round_number}: score {len(mapping) - BLOCK_SIZE} blocks")
        scores = compute_block_scores(model, samples, args.max_seq_length)
        ranking = sorted(range(len(scores)), key=lambda i: (scores[i], i))
        candidates = [{"rank": rank, "current_start": start,
                       "current_end": start + BLOCK_SIZE - 1,
                       "current_indices": list(range(start, start + BLOCK_SIZE)),
                       "original_indices": mapping[start:start + BLOCK_SIZE],
                       "score_hidden_state_indices": [start + 1, start + BLOCK_SIZE + 1],
                       "avg_distance": scores[start], "selected": rank == 1}
                      for rank, start in enumerate(ranking, 1)]
        start = ranking[0]
        removed_current = list(range(start, start + BLOCK_SIZE))
        removed_this_round = mapping[start:start + BLOCK_SIZE]
        kept = [i for i in range(len(mapping)) if i not in removed_current]
        new_mapping = [mapping[i] for i in kept]
        record = {"round": round_number, "num_layers_before": len(mapping),
                  "num_layers_after": len(new_mapping), "mapping_before": mapping[:],
                  "removed_current_indices": removed_current,
                  "removed_original_indices": removed_this_round,
                  "new_to_original_layer": new_mapping[:], "candidates": candidates}
        rebuild_layers(model, kept)
        mapping = new_mapping
        removed.extend(removed_this_round)
        log["rounds"].append(record)
        log.update(layers_removed=sorted(removed), layers_removed_in_order=removed[:],
                   layers_kept=mapping[:], new_to_original_layer=mapping[:])
        write_json(args.output_dir / f"round_{round_number:02d}.json", record)
        write_json(args.output_dir / "pruneme_log.json", log)
        print(f"Removed original layers {removed_this_round}; {len(mapping)} remain")
    if len(set(removed)) != TOTAL_PRUNE_LAYERS or len(mapping) != num_layers - TOTAL_PRUNE_LAYERS:
        raise RuntimeError("Final pruning count/mapping mismatch.")
    # Match the sources' explicit untied checkpoint serialization.
    if model.config.tie_word_embeddings:
        model.lm_head.weight = torch.nn.Parameter(model.model.embed_tokens.weight.detach().clone())
        model.config.tie_word_embeddings = False
        model.model.config.tie_word_embeddings = False
    log["status"] = "saving_model"
    write_json(args.output_dir / "pruneme_log.json", log)
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    log["status"] = "complete"
    write_json(args.output_dir / "pruneme_log.json", log)
    print(f"Saved model, tokenizer and logs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()