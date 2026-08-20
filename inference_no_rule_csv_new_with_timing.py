#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
import sys
import time
from collections import OrderedDict
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

import torch
#from peft import PeftModel                 # 未來用到qlora再回來看
try:
    from peft import PeftModel
except ImportError:
    PeftModel = None
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dxf2laserfill.converter_no_rule import BatchConverter

BASE_MODEL_NAME = "ibm-granite/granite-3b-code-instruct-128k"

MODEL_OR_ADAPTER_DIR = PROJECT_ROOT / "granite3b-pruneme-healed-ep2"

INFERENCE_INPUT_DIR = PROJECT_ROOT / "inference_input"
INFERENCE_OUTPUT_DIR = PROJECT_ROOT / "ep2_output"
INFERENCE_PATH_JSON_DIR = INFERENCE_OUTPUT_DIR / "path_json_no_rule"
TIMING_CSV_PATH = PROJECT_ROOT / "ep2_timing.cvs"

PARAMETERS_CSV_PATH = Path("/workspace/test_data/parameters_test.csv") # 只要改這個其他vllm沒有import
CSV_FILENAME_COL = "filename"
CSV_LINE_DISTANCE_COL = "line_distance"
CSV_LASER_POWER_COL = "laser_power_s"
CSV_LASER_FREQ_COL = "laser_freq"
CSV_SCAN_SPEED_COL = "scan_speed_f"

FLATTEN_DISTANCE = 0.01
MIN_CURVE_SEGMENTS = 8
SNAP_TOLERANCE = 1e-4
START_OFFSET_RATIO = 0.5
SERPENTINE = True
FORCE_CLOSE_OPEN_CONTOURS = True

SYSTEM_PROMPT = (
    "你是雷射CAM G-code生成器。請根據加工參數與路徑資料輸出正確的G-code。"
    "G-code 合法格式必須為 G0、M3、G1、M5 四行一組。"
    "只有第一個 G1 顯示 S、Q、F，之後每一行 G1 都不要顯示 S、Q、F。"
    "X/Y座標固定4位小數，S固定2位小數，Q與F為整數。不可產生零長度G1。"
)

STRICT_G0_RE = re.compile(r"^G0\s+X(?P<x>-?\d+(?:\.\d+)?)\s+Y(?P<y>-?\d+(?:\.\d+)?)$")
STRICT_G1_RE = re.compile(
    r"^G1\s+X(?P<x>-?\d+(?:\.\d+)?)\s+Y(?P<y>-?\d+(?:\.\d+)?)"
    r"(?:\s+S(?P<s>-?\d+(?:\.\d+)?))?"
    r"(?:\s+Q(?P<q>-?\d+(?:\.\d+)?))?"
    r"(?:\s+F(?P<f>-?\d+(?:\.\d+)?))?$"
)


def fmt_xy(value: str | float | Decimal) -> str:
    return format(Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP), ".4f")



def fmt_s(value: str | float | Decimal) -> str:
    return format(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP), ".2f")



def fmt_int(value: str | float | int) -> str:
    return str(int(round(float(value))))



def fmt_sec(value: float) -> str:
    return f"{value:.6f}"



def sync_device() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()



def build_converter() -> BatchConverter:
    return BatchConverter(
        flatten_distance=FLATTEN_DISTANCE,
        min_curve_segments=MIN_CURVE_SEGMENTS,
        snap_tolerance=SNAP_TOLERANCE,
        start_offset_ratio=START_OFFSET_RATIO,
        serpentine=SERPENTINE,
        force_close_open_contours=FORCE_CLOSE_OPEN_CONTOURS,
    )



def load_parameter_table(csv_path: Path) -> dict[str, dict[str, float | int]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到參數 CSV：{csv_path}")

    table: dict[str, dict[str, float | int]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            CSV_FILENAME_COL,
            CSV_LINE_DISTANCE_COL,
            CSV_LASER_POWER_COL,
            CSV_LASER_FREQ_COL,
            CSV_SCAN_SPEED_COL,
        }
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                "CSV 欄位不完整，至少需要："
                f"{CSV_FILENAME_COL}, {CSV_LINE_DISTANCE_COL}, {CSV_LASER_POWER_COL}, "
                f"{CSV_LASER_FREQ_COL}, {CSV_SCAN_SPEED_COL}"
            )

        for row in reader:
            name = (row.get(CSV_FILENAME_COL) or "").strip()
            if not name:
                continue

            table[name] = {
                "line_distance": float(row[CSV_LINE_DISTANCE_COL]),
                "laser_power": float(row[CSV_LASER_POWER_COL]),
                "laser_freq": int(round(float(row[CSV_LASER_FREQ_COL]))),
                "scan_speed": int(round(float(row[CSV_SCAN_SPEED_COL]))),
            }

    if not table:
        raise ValueError(f"CSV 內沒有可用參數資料：{csv_path}")

    return table



def lookup_params_for_dxf(
    dxf_file: Path,
    parameter_table: dict[str, dict[str, float | int]],
) -> dict[str, float | int]:
    candidates = [dxf_file.stem, dxf_file.name]
    for key in candidates:
        if key in parameter_table:
            return parameter_table[key]
    raise KeyError(f"CSV 找不到對應參數：{dxf_file.name}（預期 filename 欄有 {dxf_file.stem}）")



def group_segments_to_path_json(segments: list[Any], *, bbox: tuple[float, float, float, float]) -> dict[str, Any]:
    row_map: OrderedDict[int, dict[str, Any]] = OrderedDict()

    for seg in segments:
        row_idx = int(seg.row_index)
        if row_idx not in row_map:
            row_map[row_idx] = {
                "r": row_idx,
                "y": fmt_xy(seg.y_start),
                "s": [],
            }
        row_map[row_idx]["s"].append([fmt_xy(seg.x_start), fmt_xy(seg.x_end)])

    rows = list(row_map.values())
    min_x, min_y, max_x, max_y = bbox

    return {
        "unit": "mm",
        "mode": "laser_fill",
        "serpentine": SERPENTINE,
        "rows": rows,
        "bbox": [fmt_xy(min_x), fmt_xy(min_y), fmt_xy(max_x), fmt_xy(max_y)],
    }



def build_user_prompt(
    *,
    line_distance: float,
    laser_power: float,
    laser_freq: int,
    scan_speed: int,
    path_json: dict[str, Any],
) -> str:
    bbox = path_json["bbox"]
    rows = path_json["rows"]

    path_body = {
        "unit": path_json["unit"],
        "mode": path_json["mode"],
        "serpentine": path_json["serpentine"],
        "rows": rows,
    }

    return (
        "[參數]\n"
        f"ld={fmt_xy(line_distance)}\n"
        f"S={fmt_s(laser_power)}\n"
        f"Q={fmt_int(laser_freq)}\n"
        f"F={fmt_int(scan_speed)}\n\n"
        "[摘要]\n"
        "unit=mm\n"
        f"rows={len(rows)}\n"
        f"bbox=[{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}]\n\n"
        "[path]\n"
        + json.dumps(path_body, ensure_ascii=False, separators=(",", ":"))
    )



def estimate_max_new_tokens(path_json: dict[str, Any]) -> int:
    seg_count = sum(len(row["s"]) for row in path_json["rows"])
    estimate = seg_count * 72 + 512
    return max(8192, min(65536, estimate))



def load_model_and_tokenizer(model_or_adapter_dir: Path):
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else None

    tokenizer_source = str(model_or_adapter_dir) if (model_or_adapter_dir / "tokenizer_config.json").exists() else BASE_MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    common_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "device_map": device_map,
    }

    loaded_base = False
    if torch.cuda.is_available():
        try:
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_NAME if (model_or_adapter_dir / "adapter_config.json").exists() else str(model_or_adapter_dir),
                attn_implementation="sdpa",
                **common_kwargs,
            )
            loaded_base = True
        except Exception:
            loaded_base = False

    if not loaded_base:
        if (model_or_adapter_dir / "adapter_config.json").exists():
            model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, **common_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(str(model_or_adapter_dir), **common_kwargs)

    if (model_or_adapter_dir / "adapter_config.json").exists():
        # if PeftModel is None:
        #     raise ImportError(
        #         "PEFT is required for adapter-based inference. "
        #         "Please install peft before using this function."
        #     )
        model = PeftModel.from_pretrained(model, str(model_or_adapter_dir))
        if hasattr(model, "merge_and_unload"):
            try:
                model = model.merge_and_unload()
            except Exception:
                pass

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model.eval()
    return model, tokenizer



def clean_generation_prefix(text: str) -> str:
    text = text.replace("\r\n", "\n").strip()
    text = re.sub(r"^```(?:gcode)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    if "G21" in text:
        text = text[text.find("G21"):]
    elif "G90" in text:
        text = text[text.find("G90"):]

    return text



def normalize_no_rule_gcode_keep_model_params(
    raw_text: str,
    *,
    fallback_s: float,
    fallback_q: int,
    fallback_f: int,
    expected_segments: int,
) -> tuple[str, dict[str, str]]:
    text = clean_generation_prefix(raw_text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    filtered: list[str] = []
    for line in lines:
        if line in {"G21", "G90", "M3", "M5"}:
            filtered.append(line)
            continue
        if STRICT_G0_RE.fullmatch(line) or STRICT_G1_RE.fullmatch(line):
            filtered.append(line)

    out_lines = ["G21", "G90"]
    i = 0
    segment_count = 0

    chosen_s: str | None = None
    chosen_q: str | None = None
    chosen_f: str | None = None

    while i + 3 < len(filtered) and segment_count < expected_segments:
        g0_line = filtered[i]
        m3_line = filtered[i + 1]
        g1_line = filtered[i + 2]
        m5_line = filtered[i + 3]

        g0_match = STRICT_G0_RE.fullmatch(g0_line)
        g1_match = STRICT_G1_RE.fullmatch(g1_line)

        if not g0_match or m3_line != "M3" or not g1_match or m5_line != "M5":
            i += 1
            continue

        x0 = fmt_xy(g0_match.group("x"))
        y0 = fmt_xy(g0_match.group("y"))
        x1 = fmt_xy(g1_match.group("x"))
        y1 = fmt_xy(g1_match.group("y"))

        if x0 == x1 and y0 == y1:
            i += 4
            continue

        if chosen_s is None:
            s_raw = g1_match.group("s")
            q_raw = g1_match.group("q")
            f_raw = g1_match.group("f")

            chosen_s = fmt_s(s_raw) if s_raw is not None else fmt_s(fallback_s)
            chosen_q = fmt_int(q_raw) if q_raw is not None else fmt_int(fallback_q)
            chosen_f = fmt_int(f_raw) if f_raw is not None else fmt_int(fallback_f)

        out_lines.append(f"G0 X{x0} Y{y0}")
        out_lines.append("M3")
        if segment_count == 0:
            out_lines.append(
                f"G1 X{x1} Y{y1} "
                f"S{chosen_s} "
                f"Q{chosen_q} "
                f"F{chosen_f}"
            )
        else:
            out_lines.append(f"G1 X{x1} Y{y1}")
        out_lines.append("M5")

        segment_count += 1
        i += 4

    if chosen_s is None:
        chosen_s = fmt_s(fallback_s)
        chosen_q = fmt_int(fallback_q)
        chosen_f = fmt_int(fallback_f)

    params = {"S": chosen_s, "Q": chosen_q, "F": chosen_f}
    return "\n".join(out_lines).strip() + ("\n" if out_lines else ""), params

def generate_raw_text_with_timing(
    model: Any,
    tokenizer: Any,
    *,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int,
) -> tuple[str, float, float]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt_text, return_tensors="pt")
    if torch.cuda.is_available():
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")

    generated_token_ids: list[int] = []
    first_token_latency_sec: float | None = None
    past_key_values = None
    eos_token_id = tokenizer.eos_token_id

    sync_device()
    generation_start = time.perf_counter()

    with torch.inference_mode():
        for step in range(max_new_tokens):
            if step == 0:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True,
                )
            else:
                next_input_ids = torch.tensor([[generated_token_ids[-1]]], device=input_ids.device)
                outputs = model(
                    input_ids=next_input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )

            logits = outputs.logits[:, -1, :]
            next_token_id = int(torch.argmax(logits, dim=-1).item())
            generated_token_ids.append(next_token_id)
            past_key_values = outputs.past_key_values

            if attention_mask is not None:
                next_attention = torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, next_attention], dim=1)

            if first_token_latency_sec is None:
                sync_device()
                first_token_latency_sec = time.perf_counter() - generation_start

            if eos_token_id is not None and next_token_id == eos_token_id:
                break

    sync_device()
    inference_time_sec = time.perf_counter() - generation_start

    if first_token_latency_sec is None:
        first_token_latency_sec = 0.0

    raw_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    return raw_text, first_token_latency_sec, inference_time_sec



def main() -> int:
    INFERENCE_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    INFERENCE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    INFERENCE_PATH_JSON_DIR.mkdir(parents=True, exist_ok=True)
    RAW_OUTPUT_DIR = INFERENCE_OUTPUT_DIR / "raw_outputs"
    RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dxf_files = sorted(INFERENCE_INPUT_DIR.glob("*.dxf"))
    if not dxf_files:
        print(f"請先把 DXF 檔放進資料夾：{INFERENCE_INPUT_DIR}")
        return 1

    if not MODEL_OR_ADAPTER_DIR.exists():
        raise FileNotFoundError(f"找不到模型/Adapter資料夾：{MODEL_OR_ADAPTER_DIR}")

    parameter_table = load_parameter_table(PARAMETERS_CSV_PATH)
    model, tokenizer = load_model_and_tokenizer(MODEL_OR_ADAPTER_DIR)
    converter = build_converter()

    success = 0
    failed = 0
    timing_rows: list[dict[str, str]] = []

    for dxf_file in dxf_files:
        first_token_latency_sec = ""
        inference_time_sec = ""

        try:
            params = lookup_params_for_dxf(dxf_file, parameter_table)
            line_distance = float(params["line_distance"])
            laser_power = float(params["laser_power"])
            laser_freq = int(params["laser_freq"])
            scan_speed = int(params["scan_speed"])

            area, _report = converter.extractor.load_fill_area(dxf_file)
            bbox = area.bounds
            segments = converter._generate_fill_segments(area, line_distance)
            if not segments:
                raise ValueError("無法從此 DXF 產生任何雷射掃描線。")

            path_json = group_segments_to_path_json(segments, bbox=bbox)
            prompt_text = build_user_prompt(
                line_distance=line_distance,
                laser_power=laser_power,
                laser_freq=laser_freq,
                scan_speed=scan_speed,
                path_json=path_json,
            )

            expected_segments = sum(len(row["s"]) for row in path_json["rows"])
            max_new_tokens = estimate_max_new_tokens(path_json)

            raw_text, first_token_latency, inference_time = generate_raw_text_with_timing(
                model,
                tokenizer,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=prompt_text,
                max_new_tokens=max_new_tokens,
            )
            first_token_latency_sec = fmt_sec(first_token_latency)
            inference_time_sec = fmt_sec(inference_time)

            # Save raw model output BEFORE post-processing (for debugging)
            (RAW_OUTPUT_DIR / f"{dxf_file.stem}.txt").write_text(raw_text, encoding="utf-8")

            gcode_text, _model_params = normalize_no_rule_gcode_keep_model_params(
                raw_text,
                fallback_s=laser_power,
                fallback_q=laser_freq,
                fallback_f=scan_speed,
                expected_segments=expected_segments,
            )

            if not gcode_text.strip():
                raise ValueError("模型未產生可用的 G-code。")

            (INFERENCE_PATH_JSON_DIR / f"{dxf_file.stem}.json").write_text(
                json.dumps(path_json, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            (INFERENCE_OUTPUT_DIR / f"{dxf_file.stem}.gcode").write_text(gcode_text, encoding="utf-8")

            success += 1
            print(
                f"[OK] {dxf_file.name} | "
                f"ld={fmt_xy(line_distance)} | "
                f"S={fmt_s(laser_power)} | "
                f"Q={fmt_int(laser_freq)} | "
                f"F={fmt_int(scan_speed)} | "
                f"first_token_latency={first_token_latency_sec}s | "
                f"inference_time={inference_time_sec}s"
            )
        except Exception as e:
            failed += 1
            print(f"[FAIL] {dxf_file.name} | {e}")
        finally:
            timing_rows.append(
                {
                    "filename": dxf_file.name,
                    "first_token_latency_sec": first_token_latency_sec,
                    "inference_time_sec": inference_time_sec,
                }
            )

    with TIMING_CSV_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "first_token_latency_sec", "inference_time_sec"],
        )
        writer.writeheader()
        writer.writerows(timing_rows)

    print("=== 推論完成 ===")
    print(f"成功: {success}")
    print(f"失敗: {failed}")
    print(f"timing.csv: {TIMING_CSV_PATH}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

