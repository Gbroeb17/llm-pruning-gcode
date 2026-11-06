#!/usr/bin/env python3
"""Run in the original project environment; no model loading or inference.

Creates a new output folder. Original raw/JSON/G-code files remain untouched.
IoU and PNG require the original evaluation/plotting program.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import math
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path('/media/ssd2/llm-pruning/cs_115522107/llm-pruning-gcode')
NUMBER = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)'
WORD = re.compile(r'([A-Z])\s*(' + NUMBER + r')')


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def validate(text, count, expected_params):
    """Require explicit absolute XY blocks, allowing modal S/Q/F values.

    Fail closed on unsupported commands rather than silently removing them.
    Returns the original block lines, preserving numeric text and parameters.
    """
    lines = []
    headers = []
    for raw in text.splitlines():
        line = re.sub(r'\([^)]*\)', '', raw.split(';', 1)[0]).strip().upper()
        if not line:
            continue
        words = WORD.findall(line)
        if WORD.sub('', line).strip() or not words:
            raise ValueError(f'Unsupported content: {raw!r}')
        vals = {}
        for key, value in words:
            if key in vals:
                raise ValueError(f'Duplicate word: {raw!r}')
            vals[key] = float(value)
        if not all(math.isfinite(v) for v in vals.values()):
            raise ValueError('Non-finite number')
        if vals in ({'G': 21.0}, {'G': 90.0}):
            if lines:
                raise ValueError('Header inside motion blocks')
            headers.append(int(vals['G']))
        else:
            lines.append((line, vals))
    if headers != [21, 90]:
        raise ValueError(f'Expected G21 then G90, got {headers}')
    # A redundant final laser-off is accepted, but never a terminal motion.
    while len(lines) > 4 * count and lines[-1][1] == {'M': 5.0}:
        lines.pop()
    if len(lines) != 4 * count:
        raise ValueError(f'Expected {count} four-line blocks; got {len(lines)} lines')
    modal = {}
    observed = []
    for i in range(count):
        block = lines[4*i:4*i+4]
        for index, (line, vals) in enumerate(block):
            command = [('G', 0), ('M', 3), ('G', 1), ('M', 5)][index]
            allowed = ({'G', 'X', 'Y'} if index == 0 else
                       {'M', 'S', 'Q', 'F'} if index == 1 else
                       {'G', 'X', 'Y', 'S', 'Q', 'F'} if index == 2 else {'M'})
            if vals.get(command[0]) != command[1] or set(vals) - allowed:
                raise ValueError(f'Block {i+1}: unexpected command {line}')
            if index in (0, 2) and not {'X', 'Y'} <= vals.keys():
                raise ValueError(f'Block {i+1}: missing explicit X/Y')
            modal.update({k: vals[k] for k in ('S', 'Q', 'F') if k in vals})
            if index == 2:
                if set(modal) != {'S', 'Q', 'F'}:
                    raise ValueError(f'Block {i+1}: missing S/Q/F before cutting')
                for k, expected in expected_params.items():
                    if not math.isclose(modal[k], expected, rel_tol=0, abs_tol=1e-6):
                        raise ValueError(f'Block {i+1}: model {k}={modal[k]} differs from reference {expected}; preserved normalized file for inspection')
                observed.append(dict(modal))
    return [line for line, _ in lines], observed


def merge_main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', type=Path, default=ROOT)
    p.add_argument('--chunks-dir', type=Path)
    p.add_argument('--dxf', type=Path, default=Path('/media/ssd2/llm-pruning/shared/datasets/test_data/inference_input/test7.dxf'))
    p.add_argument('--parameters-csv', type=Path)
    p.add_argument('--original-gcode', type=Path)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    chunks = a.chunks_dir or a.project_root / 'test7_chunk_diagnostics/test7_20261026_013219_104125'
    out = a.out or a.project_root / 'test7_merged' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    required = [chunks / 'full_path_json.json', chunks / 'parameters.json', chunks / 'summary.json', a.dxf]
    specs = [('rows_000_100', 0, 100), ('rows_100_200', 100, 200), ('rows_200_290', 200, 290)]
    for name, _, _ in specs:
        required += [chunks / name / 'raw.txt', chunks / name / 'path_json.json']
    for file in required:
        if not file.is_file():
            p.error(f'Missing file: {file}')
    sys.path.insert(0, str(a.project_root / 'src'))
    sys.path.insert(0, str(a.project_root))
    mod = importlib.import_module('inference_no_rule_csv_new_with_timing')
    full = json.loads((chunks / 'full_path_json.json').read_text(encoding='utf-8-sig'))
    summary = json.loads((chunks / 'summary.json').read_text(encoding='utf-8-sig'))
    if Path(summary.get('model', '')).name != 'pruneme-skip16-qlora-ep2':
        raise ValueError('summary.json does not confirm pruneme-skip16-qlora-ep2')
    csv_path = a.parameters_csv or Path(mod.PARAMETERS_CSV_PATH)
    params = mod.lookup_params_for_dxf(a.dxf, mod.load_parameter_table(csv_path))
    expected = dict(S=float(params['laser_power']), Q=float(params['laser_freq']), F=float(params['scan_speed']))
    if len(full['rows']) != 290:
        raise ValueError('Full reference does not contain exactly 290 rows')
    out.mkdir(parents=True, exist_ok=False)
    ref = out / 'references'
    ref.mkdir()
    for file in required:
        relative = file.relative_to(chunks) if file.is_relative_to(chunks) else Path(file.name)
        target = ref / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, target)
    shutil.copy2(csv_path, ref / ('parameters_source' + csv_path.suffix))
    manifest = {'model': summary['model'], 'dxf': str(a.dxf.resolve()),
                'parameters_csv': str(csv_path.resolve()), 'expected_parameters': expected,
                'evaluation': 'NOT RUN: original evaluator/plotter required', 'chunks': []}
    save(out / 'manifest.json', manifest)
    joined = ['G21', 'G90']
    for name, start, stop in specs:
        folder = chunks / name
        path = json.loads((folder / 'path_json.json').read_text(encoding='utf-8-sig'))
        if path['rows'] != full['rows'][start:stop]:
            raise ValueError(f'{name}: rows differ from full reference slice')
        count = sum(len(row['s']) for row in path['rows'])
        if count != stop - start:
            raise ValueError(f'{name}: expected {stop-start} segments, got {count}')
        raw = (folder / 'raw.txt').read_text(encoding='utf-8-sig')
        normalized, model_params = mod.normalize_no_rule_gcode_keep_model_params(
            raw, fallback_s=expected['S'], fallback_q=expected['Q'],
            fallback_f=expected['F'], expected_segments=count)
        (out / f'{name}.gcode').write_text(normalized, encoding='utf-8')
        body, observed = validate(normalized, count, expected)
        joined.extend(body)
        manifest['chunks'].append({'name': name, 'expected_segments': count,
                                   'raw_sha256': hashlib.sha256(raw.encode()).hexdigest(),
                                   'model_params': model_params, 'cut_parameters': observed[0]})
    merged = '\n'.join(joined) + '\n'
    validate(merged, 290, expected)
    (out / 'test7.gcode').write_text(merged, encoding='utf-8')
    if a.original_gcode:
        if not a.original_gcode.is_file():
            raise FileNotFoundError(a.original_gcode)
        original = out / 'original_290rows'
        original.mkdir()
        shutil.copy2(a.original_gcode, original / 'test7.gcode')
        manifest['original_gcode'] = str(a.original_gcode.resolve())
    manifest['merged_gcode'] = str(out / 'test7.gcode')
    manifest['segments'] = 290
    manifest['validation_scope'] = 'Command format, segment count, S/Q/F and reference slices; geometry IoU not evaluated'
    save(out / 'manifest.json', manifest)
    print(f'Merged 290 segments: {out / "test7.gcode"}')
    print('Original DXF and JSON references copied to references/.')
    return 0




import argparse
import csv
import importlib.util
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path('/media/ssd2/llm-pruning/cs_115522107/llm-pruning-gcode')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def evaluate_main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--merged-gcode', required=True, type=Path)
    p.add_argument('--original-gcode', type=Path)
    p.add_argument('--project-root', type=Path, default=ROOT)
    p.add_argument('--evaluator-file', type=Path, help='Explicit evaluation.py or evaluator.py path')
    p.add_argument('--dxf', type=Path, default=Path('/media/ssd2/llm-pruning/shared/datasets/test_data/inference_input/test7.dxf'))
    p.add_argument('--param-csv', type=Path, default=Path('/media/ssd2/llm-pruning/shared/datasets/test_data/parameters_test.csv'))
    p.add_argument('--out', type=Path)
    p.add_argument('--force-close-dxf', action=argparse.BooleanOptionalAction, default=True)
    a = p.parse_args()
    root = a.project_root.resolve()
    sys.path.insert(0, str(root / 'src'))
    sys.path.insert(0, str(root))
    evaluator_file = a.evaluator_file
    if evaluator_file is None:
        candidates = [x for x in (root/'evaluation.py', root/'evaluator.py', root/'src/evaluation.py', root/'src/evaluator.py') if x.is_file()]
        if len(candidates) != 1:
            p.error('Please specify --evaluator-file with the evaluation.py supplied in chat')
        evaluator_file = candidates[0]
    for file in [a.merged_gcode, a.dxf, a.param_csv, evaluator_file, root/'run_evaluation.py'] + ([a.original_gcode] if a.original_gcode else []):
        if not file.is_file():
            p.error(f'Missing file: {file}')
    # run_evaluation imports "evaluator"; explicitly bind the user's chosen file.
    evaluator = load('evaluator', evaluator_file.resolve())
    try:
        runner = load('_test7_project_runner', root/'run_evaluation.py')
    except ModuleNotFoundError as exc:
        raise SystemExit(f'Missing project dependency: {exc}. PNG requires overlay_dxf_gcode.py and its dependencies.') from exc
    params = evaluator.load_param_csv(a.param_csv)
    if 'test7' not in params or params['test7'].get('line_distance', 0) <= 0:
        raise ValueError('Parameter CSV must contain test7 S/Q/F and positive line_distance; refusing per-output width inference')
    out = a.out or root/'test7_comparison'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    dxf_dir = out/'reference_dxf'
    dxf_dir.mkdir()
    shutil.copy2(a.dxf, dxf_dir/'test7.dxf')
    shutil.copy2(a.param_csv, out/'parameters.csv')
    settings = runner.build_argparser().parse_args([])
    settings.force_close_dxf = a.force_close_dxf
    settings.tol_mm = 0.005
    settings.path_width_mm = None
    eval_dxf_dir = dxf_dir
    if a.force_close_dxf:
        # Keep repaired DXF for reproducibility; use the exact project helper.
        eval_dxf_dir = runner._prepare_force_closed_dxf_dir(dxf_dir=dxf_dir, temp_root=out, args=settings)
    variants = [('chunked_100_100_90', a.merged_gcode)]
    if a.original_gcode:
        variants.append(('original_290rows', a.original_gcode))
    rows = []
    for label, source in variants:
        folder = out/label
        gcode_dir = folder/'gcode'
        gcode_dir.mkdir(parents=True)
        shutil.copy2(source, gcode_dir/'test7.gcode')
        metrics_csv, summary_json = evaluator.evaluate_all(
            dxf_dir=eval_dxf_dir, gcode_dir=gcode_dir, param_csv=a.param_csv,
            timing_csv=None, output_dir=folder/'metrics', tol_mm=0.005,
            path_width_mm=None)
        summary = json.loads(summary_json.read_text(encoding='utf-8'))
        if summary['num_success'] != 1 or summary['num_failures']:
            raise RuntimeError(f'{label}: evaluation failed; inspect {summary_json}')
        with metrics_csv.open(encoding='utf-8-sig', newline='') as f:
            metric = next(csv.DictReader(f))
        row = {'variant': label, **metric}
        # No measured timing is available for this comparison.
        row['inference_time'] = ''
        row['first_token_latency'] = ''
        rows.append(row)
        runner.run_overlay_batch(dxf_dir=eval_dxf_dir, gcode_dir=gcode_dir, out_dir=folder/'png')
        if not list((folder/'png').rglob('*.png')):
            raise RuntimeError(f'{label}: overlay routine did not create a PNG')
        print(f'{label}: IoU={metric["iou"]}; PNG directory={folder / "png"}')
    with (out/'comparison.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {'tol_mm': 0.005, 'force_close_dxf': a.force_close_dxf,
                'close_tol_mm': settings.close_tol_mm, 'flatten_distance': settings.flatten_distance,
                'min_curve_segments': settings.min_curve_segments,
                'evaluator_file': str(evaluator_file.resolve()),
                'merged_source': str(a.merged_gcode.resolve()),
                'original_source': str(a.original_gcode.resolve()) if a.original_gcode else None,
                'path_width': 'parameter CSV line_distance',
                'iou_alignment': 'original evaluator best Y-offset alignment',
                'timing': 'unavailable; not copied from full generation', 'results': rows}
    (out/'comparison.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Comparison table: {out / "comparison.csv"}')
    return 0




def main():
    p = argparse.ArgumentParser(description='Normalize and merge test7 chunks, evaluate IoU, and create PNG overlays.')
    p.add_argument('--project-root', type=Path, default=ROOT)
    p.add_argument('--chunks-dir', type=Path)
    p.add_argument('--dxf', type=Path, default=Path('/media/ssd2/llm-pruning/shared/datasets/test_data/inference_input/test7.dxf'))
    p.add_argument('--param-csv', type=Path, default=Path('/media/ssd2/llm-pruning/shared/datasets/test_data/parameters_test.csv'))
    p.add_argument('--evaluator-file', type=Path)
    p.add_argument('--original-gcode', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--force-close-dxf', action=argparse.BooleanOptionalAction, default=True)
    a = p.parse_args()
    # Resolve all user paths before passing them between stages.
    root = a.project_root.resolve()
    # Defaults for this experiment: no command-line arguments are needed.
    # evaluator.py may live in the project root or src/, as in run_evaluation.py.
    if a.evaluator_file is None:
        a.evaluator_file = root/'evaluator.py'
        if not a.evaluator_file.is_file() and (root/'src/evaluator.py').is_file():
            a.evaluator_file = root/'src/evaluator.py'
    if a.original_gcode is None:
        a.original_gcode = root/'inference_outputs/pruneme-skip16-qlora-ep2/test7.gcode'
    chunks = (a.chunks_dir or root/'test7_chunk_diagnostics/test7_20261026_013219_104125').resolve()
    out = (a.out or root/'test7_merge_evaluation'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
    if out.exists():
        p.error(f'Output already exists; choose a new folder: {out}')
    # Check files before starting; missing optional project dependencies are reported on import.
    required = [root/'run_evaluation.py', a.dxf, a.param_csv]
    if a.original_gcode:
        required.append(a.original_gcode)
    if a.evaluator_file:
        required.append(a.evaluator_file)
    for file in required:
        if not file.is_file():
            p.error(f'Missing file: {file}')
    saved_argv = sys.argv[:]
    try:
        sys.argv = [saved_argv[0], '--project-root', str(root), '--chunks-dir', str(chunks),
                    '--dxf', str(a.dxf.resolve()), '--parameters-csv', str(a.param_csv.resolve()),
                    '--out', str(out/'merged')]
        print('[1/2] Normalize and merge 100 + 100 + 90 segments')
        merge_main()
        sys.argv = [saved_argv[0], '--project-root', str(root),
                    '--merged-gcode', str(out/'merged/test7.gcode'),
                    '--dxf', str(a.dxf.resolve()), '--param-csv', str(a.param_csv.resolve()),
                    '--out', str(out/'evaluation'),
                    '--force-close-dxf' if a.force_close_dxf else '--no-force-close-dxf']
        if a.evaluator_file:
            sys.argv += ['--evaluator-file', str(a.evaluator_file.resolve())]
        if a.original_gcode:
            sys.argv += ['--original-gcode', str(a.original_gcode.resolve())]
        print('[2/2] Evaluate IoU and generate PNG overlays')
        evaluate_main()
    finally:
        sys.argv = saved_argv
    print(f'All outputs: {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
