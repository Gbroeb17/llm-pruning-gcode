#!/usr/bin/env python3
"""Count DXF rows/segments with the original project's preprocessing only.

Place this file beside the original inference script, above src/, and run
in the project's Python environment. Only the tokenizer is loaded; no LLM
is constructed and no inference is performed. The original module must be safe to import
(model initialization must be inside its main function).

Exit codes: 0 = all files succeeded; 1 = one or more DXFs failed;
2 = setup/input/output failure. Failed counts are blank, not zero.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_INPUT = Path(
    "/media/ssd2/llm-pruning/shared/datasets/test_data/inference_input"
)
DEFAULT_OUTPUT = Path(
    "/media/ssd2/llm-pruning/cs_115522107/llm-pruning-gcode/all_test_rows_segments.csv"
)
DEFAULT_MODEL_DIR = Path(
    "/media/ssd2/llm-pruning/cs_115522107/llm-pruning-gcode/outputs/pruneme-skip16-qlora-ep2"
)
FIELDS = ["DXF", "Rows", "Segments", "Input Tokens", "Estimated Max New Tokens", "Error"]


def aligned_table(records):
    """Readable companion; CSV keeps actual numbers without padding spaces."""
    import unicodedata

    def width(s):
        return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)

    cells = [[str(r[k]).replace("\n", " ").replace("\r", " ") for k in FIELDS]
             for r in records]
    widths = [max(width(row[i]) for row in [FIELDS] + cells) for i in range(len(FIELDS))]

    def line(row, header=False):
        parts = []
        for i, value in enumerate(row):
            padding = " " * (widths[i] - width(value))
            parts.append(padding + value if not header and i in (1, 2, 3, 4)
                         else value + padding)
        return " | ".join(parts)

    return "\n".join([line(FIELDS, True), "-+-".join("-" * w for w in widths)]
                     + [line(row) for row in cells]) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT,
                        help="Directory containing *.dxf (non-recursive)")
    parser.add_argument("--output", type=Path,
                        default=DEFAULT_OUTPUT,
                        help="Single CSV output path (UTF-8 BOM)")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR,
                        help="Local tokenizer directory used by the inference script")
    args = parser.parse_args(argv)

    if not args.input_dir.is_dir():
        parser.error(f"Input directory does not exist: {args.input_dir}")
    files = sorted(
        (p for p in args.input_dir.iterdir()
         if p.is_file() and p.suffix.lower() == ".dxf"),
        key=lambda p: (p.name.casefold(), p.name),
    )
    if not files:
        parser.error(f"No *.dxf files found in: {args.input_dir}")
    output = args.output.resolve()
    if any(output == p.resolve() for p in files):
        parser.error("Output must not overwrite an input DXF")

    # Match the uploaded test7 script: src/ takes precedence; the script's
    # own directory is also on sys.path when launched normally with Python.
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    try:
        from inference_no_rule_csv_new_with_timing import (
            PARAMETERS_CSV_PATH,
            build_converter,
            load_parameter_table,
            lookup_params_for_dxf,
            group_segments_to_path_json,
            SYSTEM_PROMPT,
            build_user_prompt,
            estimate_max_new_tokens,
        )
        from transformers import AutoTokenizer
        parameter_table = load_parameter_table(PARAMETERS_CSV_PATH)
        tokenizer = AutoTokenizer.from_pretrained(
            str(args.model_dir), use_fast=True, trust_remote_code=True,
            local_files_only=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as exc:
        print(f"Setup failed: {type(exc).__name__}: {exc}\n"
              f"Place this file above src/: {SRC_DIR}\n"
              "Use the original project's Python environment.", file=sys.stderr)
        return 2

    failed = 0
    records = []
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=FIELDS
            )
            writer.writeheader()
            for index, dxf in enumerate(files, 1):
                record = dict.fromkeys(FIELDS, "")
                record["DXF"] = dxf.name
                try:
                    params = lookup_params_for_dxf(dxf, parameter_table)
                    line_distance = float(params["line_distance"])
                    if not math.isfinite(line_distance) or line_distance <= 0:
                        raise ValueError("line_distance must be finite and positive")
                    # Fresh converter per file isolates mutable state after failures.
                    converter = build_converter()
                    area, _report = converter.extractor.load_fill_area(dxf)
                    segments = converter._generate_fill_segments(area, line_distance)
                    if not segments:
                        raise ValueError("No scan segments generated")
                    path_json = group_segments_to_path_json(segments, bbox=area.bounds)
                    rows = path_json["rows"]
                    row_count = len(rows)
                    segment_count = sum(len(r["s"]) for r in rows)
                    record.update({"Rows": row_count, "Segments": segment_count})
                    record["Estimated Max New Tokens"] = estimate_max_new_tokens(path_json)
                    user_prompt = build_user_prompt(
                        line_distance=line_distance,
                        laser_power=float(params["laser_power"]),
                        laser_freq=int(params["laser_freq"]),
                        scan_speed=int(params["scan_speed"]),
                        path_json=path_json,
                    )
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ]
                    prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    # Match the uploaded test7 script exactly, including the
                    # tokenizer's default special-token handling. No truncation.
                    record["Input Tokens"] = len(tokenizer(prompt)["input_ids"])
                except Exception as exc:
                    failed += 1
                    record["Error"] = f"{type(exc).__name__}: {exc}"
                records.append(record)
                writer.writerow(record)
                handle.flush()
                status = (f"ERROR: {record['Error']}" if record["Error"] else
                          f"rows={record['Rows']}, segments={record['Segments']}, "
                          f"input_tokens={record['Input Tokens']}, "
                          f"max_new_tokens={record['Estimated Max New Tokens']}")
                print(f"[{index}/{len(files)}] {dxf.name}: {status}", flush=True)
        table = aligned_table(records)
        table_path = output.with_name(output.stem + "_aligned.txt")
        table_path.write_text(table, encoding="utf-8-sig")
        print("\n" + table)
        print(f"Aligned table: {table_path}")
    except OSError as exc:
        print(f"CSV output failed: {exc}", file=sys.stderr)
        return 2

    print(f"CSV: {output}\nFiles: {len(files)}, succeeded: {len(files) - failed}, "
          f"failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
