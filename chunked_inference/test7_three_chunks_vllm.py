#!/usr/bin/env python3
"""Place beside the original script (above src/); run in its vLLM environment.

Diagnostic outputs only: the three G-code files are NOT a joined machining job.
Original row IDs/r, bbox and unknown metadata are preserved, never guessed.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

MODEL_DIR = "/media/ssd2/llm-pruning/cs_115522107/llm-pruning-gcode/outputs/pruneme-skip16-qlora-ep2"
INFERENCE_INPUT_DIR = Path("/media/ssd2/llm-pruning/shared/datasets/test_data/inference_input")
GPU_MEM_UTIL = 0.90
MAX_MODEL_LEN_CAP = 36864
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from inference_no_rule_csv_new_with_timing import (
    SYSTEM_PROMPT, PARAMETERS_CSV_PATH, build_converter, load_parameter_table,
    lookup_params_for_dxf, group_segments_to_path_json, build_user_prompt,
    estimate_max_new_tokens, normalize_no_rule_gcode_keep_model_params,
    fmt_xy, fmt_s, fmt_int, fmt_sec,
)
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def expected_coordinates(rows):
    """Conservative schema reader; unsupported shapes are reported, not guessed."""
    result = []
    for row_index, row in enumerate(rows):
        for segment_index, s in enumerate(row["s"]):
            if isinstance(s, (list, tuple)) and len(s) == 2:
                if all(isinstance(p, (list, tuple)) and len(p) == 2 for p in s):
                    a, b = s
                elif "y" in row and all(isinstance(p, (int, float, str)) for p in s):
                    a, b = (s[0], row["y"]), (s[1], row["y"])
                else:
                    raise ValueError("Cannot interpret s as endpoints or [x_start,x_end] with row.y")
            elif isinstance(s, (list, tuple)) and len(s) == 4:
                a, b = s[:2], s[2:]
            else:
                raise ValueError("Unsupported segment schema; inspect full_path_json.json")
            points = [list(map(float, a)), list(map(float, b))]
            if not all(math.isfinite(v) for p in points for v in p):
                raise ValueError("Non-finite expected coordinate")
            result.append({"local_row": row_index, "r": row.get("r"),
                           "segment": segment_index, "start": points[0], "end": points[1]})
    return result


def gcode_segments(text):
    """Parse absolute G0/G1 XY moves; require a known start and both axes.

    This compares G1 geometry, not laser state, speed or machining safety.
    Unsupported coordinate/motion modes invalidate the diagnostic.
    """
    x = y = mode = None
    moves = []
    for line in text.splitlines():
        line = re.sub(r"\([^)]*\)", "", line.split(";", 1)[0]).upper()
        words = [(k, float(v)) for k, v in re.findall(r"([A-Z])\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))", line)]
        gs = [v for k, v in words if k == "G"]
        if any(g in (2, 3, 20, 28, 53, 91, 92) or 54 <= g <= 59.3 for g in gs):
            raise ValueError("Unsupported arc, unit, relative, offset or homing command")
        for g in gs:
            if g in (0, 1):
                mode = int(g)
        axes = {k: v for k, v in words if k in ("X", "Y")}
        if not axes:
            continue
        if mode is None:
            raise ValueError("XY move without a known G0/G1 mode")
        nx, ny = axes.get("X", x), axes.get("Y", y)
        if nx is None or ny is None:
            raise ValueError("Cannot establish both initial XY coordinates")
        if mode == 1 and x is not None and y is not None and (nx, ny) != (x, y):
            moves.append([[x, y], [nx, ny]])
        x, y = nx, ny
    return moves


def diagnose(text, expected, tolerance):
    actual = gcode_segments(text)
    unused = set(range(len(actual)))
    matches, missing = [], []
    for i, item in enumerate(expected):
        target = [item["start"], item["end"]]
        found = None
        for j in sorted(unused):
            if all(abs(actual[j][p][k] - target[p][k]) <= tolerance for p in (0, 1) for k in (0, 1)):
                found = j
                break
        if found is None:
            missing.append({"expected_index": i, **item})
        else:
            unused.remove(found)
            matches.append({"expected_index": i, "actual_index": found})
    order = [m["actual_index"] for m in matches]
    return {"status": "ok", "tolerance": tolerance, "direction_sensitive": True,
            "scope": "G1 XY geometry only; absolute millimetres assumed; laser state not verified",
            "expected_count": len(expected), "actual_count": len(actual),
            "matched_count": len(matches), "matched_order_preserved": order == sorted(order),
            "matches": matches, "missing": missing,
            "unmatched_actual": [{"actual_index": j, "points": actual[j]} for j in sorted(unused)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-parent", type=Path, default=PROJECT_ROOT / "test7_chunk_diagnostics")
    parser.add_argument("--tolerance", type=float, default=0.001)
    args = parser.parse_args()
    if not math.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error("--tolerance must be finite and positive")
    args.output_parent.mkdir(parents=True, exist_ok=True)
    out = args.output_parent / datetime.now().strftime("test7_%Y%m%d_%H%M%S_%f")
    out.mkdir(exist_ok=False)
    print(f"!!! MODEL: {MODEL_DIR} !!!")
    print(f"New diagnostic output directory: {out}")
    warnings = [f"Using the user's fine-tuned model directory: {MODEL_DIR}.",
                "Original bbox, row r values and all extra metadata are preserved verbatim.",
                "Extra metadata may describe the full path and become inconsistent with sliced rows; no semantics are inferred.",
                "Row IDs are not rebased. If the template requires local IDs/counts, inspect the saved metadata audit before interpreting results.",
                "Three independent generations; do not concatenate these files for machining."]
    summary = {"model": MODEL_DIR, "warnings": warnings, "chunks": []}
    try:
        dxf = INFERENCE_INPUT_DIR / "test7.dxf"
        if not dxf.is_file():
            raise FileNotFoundError(dxf)
        params = lookup_params_for_dxf(dxf, load_parameter_table(PARAMETERS_CSV_PATH))
        ld, power = float(params["line_distance"]), float(params["laser_power"])
        freq, speed = int(params["laser_freq"]), int(params["scan_speed"])
        converter = build_converter()
        area, report = converter.extractor.load_fill_area(dxf)
        segments = converter._generate_fill_segments(area, ld)
        if not segments:
            raise ValueError("No scan segments generated")
        full = group_segments_to_path_json(segments, bbox=area.bounds)
        save(out / "full_path_json.json", full)
        save(out / "parameters.json", params)
        rows = full["rows"]
        if len(rows) != 290:
            raise ValueError(f"Expected exactly 290 rows for this experiment, got {len(rows)}; refusing silent omission")
        save(out / "metadata_audit.json", {
            "top_level_fields_preserved": {k: v for k, v in full.items() if k != "rows"},
            "row_fields": sorted({k for row in rows for k in row}),
            "r_values": [row.get("r") for row in rows], "warnings": warnings})
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        prepared = []
        for start, stop in ((0, 100), (100, 200), (200, 290)):
            folder = out / f"rows_{start:03d}_{stop:03d}"
            folder.mkdir()
            path = copy.deepcopy(full)
            path["rows"] = copy.deepcopy(rows[start:stop])
            save(folder / "path_json.json", path)
            user = build_user_prompt(line_distance=ld, laser_power=power, laser_freq=freq,
                                     scan_speed=speed, path_json=path)
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
            save(folder / "messages.json", messages)
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            (folder / "prompt.txt").write_text(prompt, encoding="utf-8")
            maximum = estimate_max_new_tokens(path)
            prompt_count = len(tokenizer(prompt)["input_ids"])
            meta = {"rows_slice": [start, stop], "expected_segments": sum(len(r["s"]) for r in path["rows"]),
                    "estimated_max_new_tokens": maximum, "tokenizer_prompt_tokens": prompt_count,
                    "warnings": warnings}
            save(folder / "diagnostics.json", meta)
            summary["chunks"].append(meta)
            prepared.append((folder, path, prompt, maximum, meta))
        needed = max(m[4]["tokenizer_prompt_tokens"] + m[3] for m in prepared)
        if needed > MAX_MODEL_LEN_CAP:
            raise ValueError(f"Required context {needed} exceeds cap {MAX_MODEL_LEN_CAP}; refusing to change max_tokens or truncate")
        context = min(needed + 256, MAX_MODEL_LEN_CAP)
        summary["max_model_len"] = context
        llm = LLM(model=MODEL_DIR, dtype="bfloat16", max_model_len=context,
                  gpu_memory_utilization=GPU_MEM_UTIL, trust_remote_code=True)
        failed = False
        for folder, path, prompt, maximum, meta in prepared:
            try:
                t0 = time.perf_counter()
                output = llm.generate([prompt], [SamplingParams(temperature=0.0, max_tokens=maximum)])[0]
                meta["generation_wall_sec"] = time.perf_counter() - t0
                completion = output.outputs[0]
                raw = completion.text
                (folder / "raw.txt").write_text(raw, encoding="utf-8")
                save(folder / "output_token_ids.json", list(completion.token_ids))
                meta.update(finish_reason=completion.finish_reason, stop_reason=completion.stop_reason,
                            output_tokens=len(completion.token_ids),
                            vllm_prompt_tokens=(len(output.prompt_token_ids) if output.prompt_token_ids is not None else None))
                normalized, model_params = normalize_no_rule_gcode_keep_model_params(
                    raw, fallback_s=power, fallback_q=freq, fallback_f=speed,
                    expected_segments=meta["expected_segments"])
                (folder / "normalized.gcode").write_text(normalized, encoding="utf-8")
                meta["model_params"] = model_params
                for label, text in (("raw", raw), ("normalized", normalized)):
                    try:
                        meta[f"{label}_coordinate_matching"] = diagnose(text, expected_coordinates(path["rows"]), args.tolerance)
                    except Exception as exc:
                        meta[f"{label}_coordinate_matching"] = {"status": "unavailable", "warning": str(exc)}
                if not normalized.strip():
                    raise ValueError("Normalizer returned empty G-code")
            except Exception:
                failed = True
                meta["error"] = traceback.format_exc()
            finally:
                save(folder / "diagnostics.json", meta)
                save(out / "summary.json", summary)
                print(f"{folder.name}: finish={meta.get('finish_reason')}, tokens={meta.get('output_tokens')}, error={bool(meta.get('error'))}")
        return 1 if failed else 0
    except Exception:
        summary["fatal_error"] = traceback.format_exc()
        print(summary["fatal_error"], file=sys.stderr)
        return 1
    finally:
        save(out / "summary.json", summary)
        (out / "WARNINGS.txt").write_text("\n".join(warnings), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
