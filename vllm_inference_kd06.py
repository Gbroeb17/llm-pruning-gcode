#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

#/media/ssd2/llm-pruning/cs_115522107/llm-pruning-gcode/outputs/skip16_ms16384_r16_ep2_run1
MODEL_DIR = "/media/ssd2/llm-pruning/cs_115522107/llm-pruning-gcode/experiments/multi_block_pruning/outputs/kd06_seed44-gptq2bit-multiblock-b16-n1-skip16"
ADAPTER_DIR = Path("/media/ssd2/llm-pruning/cs_115522107/llm-pruning-gcode/qlora_runs/kd06_gptq2bit_skip16_ms16384_r16_ep2_run1/final_adapter")
INFERENCE_INPUT_DIR = Path("/media/ssd2/llm-pruning/shared/datasets/test_data/inference_input")          # 1000 筆 DXF
INFERENCE_OUTPUT_DIR = Path("/media/ssd2/llm-pruning/cs_115522107/llm-pruning-gcode/inference_outputs/kd06_gptq2bit_skip16_qep2")
TIMING_CSV_PATH = Path("/media/ssd2/llm-pruning/cs_115522107/llm-pruning-gcode/inference_outputs/kd06_gptq2bit_skip16_qep2-timing.csv")

# vLLM 引擎設定
GPU_MEM_UTIL = 0.90
# max_model_len 會在下面根據實際 prompt + max_tokens 自動算,
# 但給一個硬上限避免 OOM (A100 80GB)
MAX_MODEL_LEN_CAP = 36864

# ============================================================
# Import 原腳本的所有處理函式 (完全沿用,不重寫)
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

def parse_args():
    parser = argparse.ArgumentParser(description="vLLM inference with a GPTQ 2-bit pruned base and an unmerged LoRA adapter.")
    parser.add_argument("--model_dir", type=Path, default=Path(MODEL_DIR))
    parser.add_argument("--adapter_dir", type=Path, default=ADAPTER_DIR)
    parser.add_argument("--output_dir", type=Path, default=INFERENCE_OUTPUT_DIR)
    parser.add_argument("--timing_csv", type=Path)
    parser.add_argument("--limit", type=int, help="Run only the first N DXFs for compatibility testing.")
    return parser.parse_args()


def validate_adapter(model_dir, adapter_dir):
    with (model_dir / "config.json").open(encoding="utf-8") as handle:
        base_config = json.load(handle)
    quantization = base_config.get("quantization_config", {})
    if quantization.get("quant_method") != "gptq" or quantization.get("bits") != 2:
        raise ValueError("Expected the pruned GPTQ 2-bit base checkpoint.")
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.is_file():
        raise ValueError(f"Adapter is not ready: {config_path}. Finish LoRA training or specify --adapter_dir.")
    with config_path.open(encoding="utf-8") as handle:
        adapter_config = json.load(handle)
    if adapter_config.get("peft_type") != "LORA":
        raise ValueError("Expected a PEFT LoRA adapter.")
    declared_base = adapter_config.get("base_model_name_or_path")
    if not declared_base or Path(declared_base).resolve() != model_dir.resolve():
        raise ValueError("Adapter's recorded base does not match --model_dir; use the exact pruned training base.")
    if not any((adapter_dir / name).is_file() for name in ["adapter_model.safetensors", "adapter_model.bin"]):
        raise ValueError("No trained adapter weights were found.")
    rank = adapter_config.get("r")
    if type(rank) is not int or rank <= 0:
        raise ValueError("Adapter rank must be a positive integer.")
    if adapter_config.get("rank_pattern"):
        rank = max(rank, *adapter_config["rank_pattern"].values())
    return rank


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive.")
    rank = validate_adapter(args.model_dir, args.adapter_dir)
    INFERENCE_OUTPUT_DIR = args.output_dir
    TIMING_CSV_PATH = args.timing_csv or args.output_dir.parent / f"{args.output_dir.name}-timing.csv"
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    from inference_no_rule_csv_new_with_timing import (
        SYSTEM_PROMPT, PARAMETERS_CSV_PATH, build_converter, load_parameter_table,
        lookup_params_for_dxf, group_segments_to_path_json, build_user_prompt,
        estimate_max_new_tokens, normalize_no_rule_gcode_keep_model_params, fmt_sec,
    )
    print(f"vLLM version: {vllm.__version__}")
    print(f"GPTQ 2-bit base: {args.model_dir}")
    print(f"LoRA adapter: {args.adapter_dir}; rank: {rank}")
    TIMING_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    INFERENCE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path_json_dir = INFERENCE_OUTPUT_DIR / "path_json_no_rule"
    path_json_dir.mkdir(parents=True, exist_ok=True)
    raw_output_dir = INFERENCE_OUTPUT_DIR / "raw_outputs"
    raw_output_dir.mkdir(parents=True, exist_ok=True)

    dxf_files = sorted(INFERENCE_INPUT_DIR.glob("*.dxf"))  # 調幾筆dxf測資
    if args.limit is not None:
        dxf_files = dxf_files[:args.limit]
    if not dxf_files:
        print(f"找不到 DXF: {INFERENCE_INPUT_DIR}")
        return 1
    print(f"找到 {len(dxf_files)} 筆 DXF")

    parameter_table = load_parameter_table(PARAMETERS_CSV_PATH)
    converter = build_converter()

    # tokenizer (從 ep1 目錄載入,跟原腳本一致)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ========================================================
    # 階段一: 準備所有 prompt + 每筆的 max_tokens (全用原函式)
    # ========================================================
    print("=== 階段一: 準備 prompts ===")
    prompts: list[str] = []
    sampling_params_list: list[SamplingParams] = []
    metadata: list[dict] = []   # 存每筆的 dxf / path_json / params,後處理用
    prep_failed = 0

    max_prompt_tokens = 0
    max_total_len = 0

    for dxf_file in dxf_files:
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
                raise ValueError("無法產生掃描線")

            path_json = group_segments_to_path_json(segments, bbox=bbox)
            user_prompt = build_user_prompt(
                line_distance=line_distance,
                laser_power=laser_power,
                laser_freq=laser_freq,
                scan_speed=scan_speed,
                path_json=path_json,
            )

            # chat template (跟原腳本 generate_raw_text_with_timing 一致)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # token 公式: 原封不動
            max_new = estimate_max_new_tokens(path_json)

            # 算這筆的 prompt token 數 (供 max_model_len 估算)
            prompt_tok = len(tokenizer(prompt_text)["input_ids"])
            total_len = prompt_tok + max_new
            max_prompt_tokens = max(max_prompt_tokens, prompt_tok)
            max_total_len = max(max_total_len, total_len)

            prompts.append(prompt_text)
            sampling_params_list.append(
                SamplingParams(temperature=0.0, max_tokens=max_new)  # greedy
            )
            metadata.append({
                "dxf": dxf_file,
                "path_json": path_json,
                "expected_segments": sum(len(r["s"]) for r in path_json["rows"]),
                "laser_power": laser_power,
                "laser_freq": laser_freq,
                "scan_speed": scan_speed,
                "max_new": max_new,
                "prompt_tok": prompt_tok,
            })
        except Exception as e:
            prep_failed += 1
            print(f"[PREP FAIL] {dxf_file.name}: {e}")

    print(f"準備完成: {len(prompts)} 筆 (失敗 {prep_failed})")
    if not prompts:
        print("No valid prompts remain; engine was not loaded.")
        return 1
    print(f"最長 prompt: {max_prompt_tokens} tokens")
    print(f"最長 prompt+max_tokens: {max_total_len} tokens")

    # 決定 max_model_len: 涵蓋最長的 prompt+輸出,但不超過硬上限
    needed = max_total_len
    if needed > MAX_MODEL_LEN_CAP:
        print(f"!! 警告: 最長需要 {needed} 超過上限 {MAX_MODEL_LEN_CAP}")
        print(f"   設 max_model_len={MAX_MODEL_LEN_CAP},超長那筆會被截斷")
        print(f"   (若不可接受,需提高 MAX_MODEL_LEN_CAP 但小心 OOM)")
    max_model_len = min(needed + 256, MAX_MODEL_LEN_CAP)  # 留一點 buffer
    print(f"設定 max_model_len = {max_model_len}")

    # ========================================================
    # 階段二: 載入 vLLM + 批次生成 (唯一換掉的部分)
    # ========================================================
    print("=== 階段二: 載入 vLLM ===")
    llm = LLM(
        model=str(args.model_dir),
        dtype="float16",
        quantization="humming",
        enable_lora=True,
        max_lora_rank=rank,
        max_model_len=max_model_len,
        gpu_memory_utilization=GPU_MEM_UTIL,
        trust_remote_code=True,
    )

    print("=== 開始批次生成 ===")
    t0 = time.perf_counter()
    outputs = llm.generate(
        prompts, sampling_params_list,
        lora_request=LoRARequest("kd06_healed", 1, str(args.adapter_dir)),
    )
    total_gen_time = time.perf_counter() - t0
    print(f"批次生成完成: {total_gen_time:.1f}s 共 {len(outputs)} 筆")
    print(f"平均每筆: {total_gen_time / max(1, len(outputs)):.2f}s (批次攤平,非單筆真實延遲)")

    # ========================================================
    # 階段三: 後處理每筆 (全用原函式,完全照舊)
    # ========================================================
    print("=== 階段三: 後處理 ===")
    success = 0
    failed = 0
    timing_rows: list[dict] = []
    # vLLM 批次無法量單筆延遲;用批次平均填 timing (paper 要誠實註明)
    approx_per_sample = total_gen_time / max(1, len(outputs))

    for output, meta in zip(outputs, metadata):
        dxf_file = meta["dxf"]
        try:
            raw_text = output.outputs[0].text

            # 存 raw (除錯用)
            (raw_output_dir / f"{dxf_file.stem}.txt").write_text(raw_text, encoding="utf-8")

            # 後處理: 原封不動
            gcode_text, _model_params = normalize_no_rule_gcode_keep_model_params(
                raw_text,
                fallback_s=meta["laser_power"],
                fallback_q=meta["laser_freq"],
                fallback_f=meta["scan_speed"],
                expected_segments=meta["expected_segments"],
            )

            if not gcode_text.strip():
                raise ValueError("後處理後無有效 G-code")

            (path_json_dir / f"{dxf_file.stem}.json").write_text(
                json.dumps(meta["path_json"], ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            (INFERENCE_OUTPUT_DIR / f"{dxf_file.stem}.gcode").write_text(
                gcode_text, encoding="utf-8"
            )
            success += 1
        except Exception as e:
            failed += 1
            print(f"[POST FAIL] {dxf_file.name}: {e}")
        finally:
            timing_rows.append({
                "filename": dxf_file.name,
                "first_token_latency_sec": fmt_sec(0.0),               # vLLM 批次無單筆值
                "inference_time_sec": fmt_sec(approx_per_sample),      # 批次平均近似
            })

    with TIMING_CSV_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename", "first_token_latency_sec", "inference_time_sec"]
        )
        writer.writeheader()
        writer.writerows(timing_rows)

    print("=== 完成 ===")
    print(f"成功: {success}  失敗: {failed}")
    print(f"總生成時間: {total_gen_time:.1f}s")
    print(f"輸出: {INFERENCE_OUTPUT_DIR}")
    print(f"timing: {TIMING_CSV_PATH} (注意: 單筆 timing 為批次平均近似)")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
