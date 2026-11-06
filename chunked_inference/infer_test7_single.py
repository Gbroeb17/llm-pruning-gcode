#!/usr/bin/env python3
"""Infer the complete test7.dxf once using the existing project's helpers.

Run in the same Linux Python/vLLM environment as the original inference.
The helper module must be import-safe (its batch runner guarded by __main__).
No rows are sliced, no prompts are truncated, and no chunks are generated.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path('/media/ssd2/llm-pruning/cs_115522107/llm-pruning-gcode')
MODEL = ROOT / 'outputs/pruneme-skip16-qlora-ep2'
DXF = Path('/media/ssd2/llm-pruning/shared/datasets/test_data/inference_input/test7.dxf')


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--model-dir', type=Path, default=MODEL)
    parser.add_argument('--dxf', type=Path, default=DXF)
    parser.add_argument('--parameters-csv', type=Path,
                        help='Defaults to PARAMETERS_CSV_PATH from the original helper')
    parser.add_argument('--output-parent', type=Path,
                        help='Defaults to PROJECT_ROOT/test7_single_outputs')
    parser.add_argument('--max-model-len', type=int, default=36864,
                        help='Context limit; fail rather than truncate if too small')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.90)
    parser.add_argument('--tensor-parallel-size', type=int, default=1)
    args = parser.parse_args(argv)
    if args.max_model_len <= 0 or args.tensor_parallel_size <= 0:
        parser.error('Context length and tensor parallel size must be positive')
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error('--gpu-memory-utilization must be between 0 and 1')
    for path in (args.project_root, args.model_dir):
        if not path.is_dir():
            parser.error(f'Directory not found: {path}')
    if not args.dxf.is_file():
        parser.error(f'DXF not found: {args.dxf}')
    if not (args.model_dir / 'config.json').is_file():
        parser.error('Model must be a merged checkpoint with config.json')

    # Same lookup order as the existing inference wrapper: src/ then root.
    sys.path.insert(0, str(args.project_root))
    sys.path.insert(0, str(args.project_root / 'src'))
    out_parent = args.output_parent or args.project_root / 'test7_single_outputs'
    out = out_parent / datetime.now().strftime('test7_%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    summary = {'status': 'running', 'model': str(args.model_dir),
               'dxf': str(args.dxf), 'output_dir': str(out),
               'samples': 1, 'chunking': False}
    print(f'Model: {args.model_dir}\nDXF: {args.dxf}\nOutput: {out}', flush=True)
    try:
        h = importlib.import_module('inference_no_rule_csv_new_with_timing')
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        summary['helper_file'] = h.__file__
        csv_path = args.parameters_csv or h.PARAMETERS_CSV_PATH
        summary['parameters_csv'] = str(csv_path)
        preprocess_start = time.perf_counter()
        params = h.lookup_params_for_dxf(args.dxf, h.load_parameter_table(csv_path))
        ld, power = float(params['line_distance']), float(params['laser_power'])
        freq, speed = int(params['laser_freq']), int(params['scan_speed'])
        converter = h.build_converter()
        area, report = converter.extractor.load_fill_area(args.dxf)
        segments = converter._generate_fill_segments(area, ld)
        if not segments:
            raise ValueError('DXF produced no fill segments')
        path_json = h.group_segments_to_path_json(segments, bbox=area.bounds)
        expected_segments = sum(len(row['s']) for row in path_json['rows'])
        user_prompt = h.build_user_prompt(
            line_distance=ld, laser_power=power, laser_freq=freq,
            scan_speed=speed, path_json=path_json)
        messages = [{'role': 'system', 'content': h.SYSTEM_PROMPT},
                    {'role': 'user', 'content': user_prompt}]
        tokenizer = AutoTokenizer.from_pretrained(
            str(args.model_dir), use_fast=True, trust_remote_code=True,
            local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        # Supply explicit token IDs so budget counting and vLLM input agree.
        prompt_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
        max_tokens = int(h.estimate_max_new_tokens(path_json))
        if max_tokens <= 0:
            raise ValueError('Helper returned a non-positive generation budget')
        summary.update(rows=len(path_json['rows']), segments=expected_segments,
                       prompt_tokens=len(prompt_ids), max_new_tokens=max_tokens,
                       preprocessing_sec=time.perf_counter() - preprocess_start)
        save_json(out / 'parameters.json', params)
        save_json(out / 'path_json.json', path_json)
        save_json(out / 'messages.json', messages)
        (out / 'prompt.txt').write_text(prompt, encoding='utf-8')
        required = len(prompt_ids) + max_tokens
        if required > args.max_model_len:
            raise ValueError(
                f'Full sample needs {required} tokens (prompt={len(prompt_ids)}, '
                f'generation={max_tokens}); --max-model-len={args.max_model_len}. '
                'Increase the limit only if the model and GPU support it. No truncation performed.')
        context = min(required + 256, args.max_model_len)
        summary['max_model_len'] = context
        print(f'Rows={summary["rows"]}, segments={expected_segments}, '
              f'prompt tokens={len(prompt_ids)}, max new tokens={max_tokens}', flush=True)
        t0 = time.perf_counter()
        llm = LLM(model=str(args.model_dir), dtype='bfloat16',
                  max_model_len=context,
                  gpu_memory_utilization=args.gpu_memory_utilization,
                  tensor_parallel_size=args.tensor_parallel_size,
                  trust_remote_code=True)
        summary['model_load_sec'] = time.perf_counter() - t0
        t0 = time.perf_counter()
        # Exactly one request, one completion, and one generate call.
        results = llm.generate(
            [{'prompt_token_ids': prompt_ids}],
            SamplingParams(n=1, temperature=0.0, max_tokens=max_tokens))
        summary['generation_sec'] = time.perf_counter() - t0
        completion = results[0].outputs[0]
        raw = completion.text
        (out / 'test7_raw.txt').write_text(raw, encoding='utf-8')
        summary.update(finish_reason=completion.finish_reason,
                       stop_reason=completion.stop_reason,
                       output_tokens=len(completion.token_ids))
        t0 = time.perf_counter()
        normalized, model_params = h.normalize_no_rule_gcode_keep_model_params(
            raw, fallback_s=power, fallback_q=freq, fallback_f=speed,
            expected_segments=expected_segments)
        (out / 'test7.gcode').write_text(normalized, encoding='utf-8')
        summary.update(normalization_sec=time.perf_counter() - t0,
                       model_params=model_params)
        if not normalized.strip():
            raise ValueError('Normalizer returned empty G-code; raw output retained')
        summary['status'] = 'truncated' if completion.finish_reason == 'length' else 'completed'
        print(f'Generation: {summary["generation_sec"]:.3f} s; '
              f'output tokens: {summary["output_tokens"]}; '
              f'finish: {completion.finish_reason}\nG-code: {out / "test7.gcode"}', flush=True)
        if summary['status'] == 'truncated':
            print('Generation hit its token limit; output may be incomplete.', file=sys.stderr)
            return 2
        return 0
    except Exception:
        summary['status'] = 'failed'
        summary['error'] = traceback.format_exc()
        print(summary['error'], file=sys.stderr)
        return 1
    finally:
        summary['total_sec'] = time.perf_counter() - started
        save_json(out / 'summary.json', summary)
        print(f'Total: {summary["total_sec"]:.3f} s; summary: {out / "summary.json"}', flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
