

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from dxf2laserfill.dxf_parser import (
        DXFGeometryExtractor,
        _build_fill_area,
        _force_close_linework,
        _import_ezdxf_dependencies,
        _snap_linework_endpoints,
    )
    from dxf2laserfill.models import DXFParseReport
except Exception as exc:  # deferred until force-close mode is actually used
    DXFGeometryExtractor = None  # type: ignore[assignment]
    DXFParseReport = None  # type: ignore[assignment]
    _build_fill_area = None  # type: ignore[assignment]
    _force_close_linework = None  # type: ignore[assignment]
    _import_ezdxf_dependencies = None  # type: ignore[assignment]
    _snap_linework_endpoints = None  # type: ignore[assignment]
    DXF_PARSER_IMPORT_ERROR = exc
else:
    DXF_PARSER_IMPORT_ERROR = None

from evaluator import evaluate_all
from overlay_dxf_gcode import run_overlay_batch


DEFAULT_DXF_DIR = Path("/workspace/test_data/inference_input")
DEFAULT_GCODE_DIR = Path("/workspace/llm-pruning-gcode/pruneme_only_100_inference_output")
DEFAULT_PARAM_CSV = Path("/workspace/test_data/parameters_test.csv")
DEFAULT_TIMING_CSV = Path("/workspace/llm-pruning-gcode/pruneme_only_100_timing.csv")
DEFAULT_OUTPUT_DIR = Path("/workspace/llm-pruning-gcode/pruneme_only_100_eval")
DEFAULT_OVERLAY_DIRNAME = "pruneme_only_100_overlay_png"
DEFAULT_TOL_MM = 0.005
DEFAULT_CLOSE_TOL_MM = 1e-4
DEFAULT_FLATTEN_DISTANCE = 0.01
DEFAULT_MIN_CURVE_SEGMENTS = 8
DEFAULT_FORCE_CLOSE_DXF = True


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run overlay generation and DXF-to-G-code evaluation in one command. "
            "If no arguments are given, it uses the built-in project defaults."
        )
    )
    p.add_argument("--dxf-dir", default=DEFAULT_DXF_DIR, type=Path)
    p.add_argument("--gcode-dir", default=DEFAULT_GCODE_DIR, type=Path)
    p.add_argument("--param-csv", default=DEFAULT_PARAM_CSV, type=Path)
    p.add_argument("--timing-csv", default=DEFAULT_TIMING_CSV, type=Path)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    p.add_argument("--overlay-out-dir", default=None, type=Path)
    p.add_argument("--tol-mm", type=float, default=DEFAULT_TOL_MM)
    p.add_argument("--path-width-mm", type=float, default=None)
    p.add_argument(
        "--force-close-dxf",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_FORCE_CLOSE_DXF,
        help=(
            "Before overlay/evaluation, build a temporary DXF folder using the force-close "
            "logic imported from src/dxf2laserfill/dxf_parser.py. Default: enabled. "
            "Use --no-force-close-dxf to disable."
        ),
    )
    p.add_argument(
        "--close-tol-mm",
        type=float,
        default=DEFAULT_CLOSE_TOL_MM,
        help="Endpoint snapping / force-close tolerance passed to DXFGeometryExtractor.",
    )
    p.add_argument(
        "--flatten-distance",
        type=float,
        default=DEFAULT_FLATTEN_DISTANCE,
        help="Curve flattening distance passed to DXFGeometryExtractor.",
    )
    p.add_argument(
        "--min-curve-segments",
        type=int,
        default=DEFAULT_MIN_CURVE_SEGMENTS,
        help="Minimum curve segments passed to DXFGeometryExtractor.",
    )
    p.add_argument(
        "--skip-overlay",
        action="store_true",
        help="Skip overlay PNG generation and only run evaluation.",
    )
    return p



def _resolve_input_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.is_absolute():
        return path
    return (PROJECT_DIR / path).resolve()



def _validate_inputs(dxf_dir: Path, gcode_dir: Path, param_csv: Path, timing_csv: Path) -> None:
    missing: list[str] = []
    if not dxf_dir.exists() or not dxf_dir.is_dir():
        missing.append(f"DXF directory not found: {dxf_dir}")
    if not gcode_dir.exists() or not gcode_dir.is_dir():
        missing.append(f"G-code directory not found: {gcode_dir}")
    if not param_csv.exists() or not param_csv.is_file():
        missing.append(f"Parameter CSV not found: {param_csv}")
    if not timing_csv.exists() or not timing_csv.is_file():
        missing.append(f"Timing CSV not found: {timing_csv}")
    if missing:
        raise FileNotFoundError("\n".join(missing))



def _require_project_dxf_parser() -> None:
    if DXF_PARSER_IMPORT_ERROR is None:
        return
    raise ImportError(
        "Cannot import src/dxf2laserfill/dxf_parser.py. "
        "Please make sure run_evaluation.py is placed in the project root where the src/ folder exists, "
        "or install the package first. Original import error: "
        f"{DXF_PARSER_IMPORT_ERROR}"
    ) from DXF_PARSER_IMPORT_ERROR



def _make_extractor(args: argparse.Namespace) -> Any:
    _require_project_dxf_parser()
    return DXFGeometryExtractor(  # type: ignore[misc]
        flatten_distance=args.flatten_distance,
        min_curve_segments=args.min_curve_segments,
        snap_tolerance=args.close_tol_mm,
        force_close_open_contours=True,
    )



def _linework_from_dxf_with_project_parser(dxf_path: Path, extractor: Any) -> tuple[list[Any], Any]:
    """
    Parse DXF with the same force-close logic used by dxf_parser.py.

    DXFGeometryExtractor.load_fill_area() returns only the final fill area, not the repaired linework.
    This helper therefore reuses the extractor's own entity/path conversion methods and the private
    repair helpers from dxf_parser.py, then returns the linework so it can be written to a temporary DXF.
    """
    _require_project_dxf_parser()
    ezdxf, ezpath, text2path = _import_ezdxf_dependencies()  # type: ignore[misc]

    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    report = DXFParseReport(source_file=dxf_path)  # type: ignore[misc]

    linework: list[Any] = []
    for entity in msp:
        report.entity_count += 1
        for path_obj in extractor._entity_to_paths(
            entity,
            ezpath=ezpath,
            text2path=text2path,
            report=report,
        ):
            linework.extend(extractor._path_to_linework(path_obj, report=report))

    if not linework:
        raise ValueError(f"{dxf_path.name} parsed to no usable boundary linework.")

    linework, snapped_endpoint_count = _snap_linework_endpoints(  # type: ignore[misc]
        linework,
        tolerance=extractor.snap_tolerance,
    )
    if snapped_endpoint_count > 0:
        report.warnings.append(f"已對齊 {snapped_endpoint_count} 個端點，以修正微小座標落差。")

    area = _build_fill_area(linework, tolerance=extractor.snap_tolerance)  # type: ignore[misc]
    if not area.is_empty:
        report.line_count = len(linework)
        return linework, report

    repaired_linework, added_gap_count = _force_close_linework(  # type: ignore[misc]
        linework,
        tolerance=extractor.snap_tolerance,
    )
    report.auto_closed_gap_count += added_gap_count
    if added_gap_count > 0:
        report.warnings.append(f"偵測到未封閉邊界，已自動補上 {added_gap_count} 段缺口並強制封閉。")

    area = _build_fill_area(repaired_linework, tolerance=extractor.snap_tolerance)  # type: ignore[misc]
    if area.is_empty:
        report.line_count = len(repaired_linework)
        raise ValueError(
            f"{dxf_path.name} cannot form a closed fill area even after dxf_parser.py force-close repair."
        )

    report.line_count = len(repaired_linework)
    return repaired_linework, report



def _write_linework_to_dxf(linework: list[Any], dst_path: Path) -> None:
    """Write repaired linework to a temporary DXF for overlay/evaluation use."""
    _require_project_dxf_parser()
    ezdxf, _, _ = _import_ezdxf_dependencies()  # type: ignore[misc]

    doc = ezdxf.new("R2000")
    msp = doc.modelspace()

    for line in linework:
        coords = list(line.coords)
        if len(coords) < 2:
            continue
        for start, end in zip(coords, coords[1:]):
            msp.add_line(
                (float(start[0]), float(start[1])),
                (float(end[0]), float(end[1])),
                dxfattribs={"layer": "FORCE_CLOSED_FROM_DXF_PARSER"},
            )

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(dst_path)



def _prepare_force_closed_dxf_dir(
    *,
    dxf_dir: Path,
    temp_root: Path,
    args: argparse.Namespace,
) -> Path:
    """
    Build a temporary DXF folder using dxf_parser.py's force-close logic.

    Original DXF files are not modified. Temporary DXFs are deleted automatically when
    the program exits.
    """
    extractor = _make_extractor(args)
    closed_dxf_dir = temp_root / "force_closed_dxf"
    closed_dxf_dir.mkdir(parents=True, exist_ok=True)

    dxf_files = sorted(dxf_dir.glob("*.dxf"), key=lambda path: path.name.lower())
    if not dxf_files:
        raise FileNotFoundError(f"No DXF files found in: {dxf_dir}")

    total_path_auto_closed = 0
    total_gap_auto_closed = 0
    total_warnings = 0

    for dxf_file in dxf_files:
        linework, report = _linework_from_dxf_with_project_parser(dxf_file, extractor)
        dst = closed_dxf_dir / dxf_file.name
        _write_linework_to_dxf(linework, dst)

        total_path_auto_closed += getattr(report, "auto_closed_path_count", 0)
        total_gap_auto_closed += getattr(report, "auto_closed_gap_count", 0)
        total_warnings += len(getattr(report, "warnings", []))

        extras = []
        if getattr(report, "auto_closed_path_count", 0):
            extras.append(f"path_auto_closed={report.auto_closed_path_count}")
        if getattr(report, "auto_closed_gap_count", 0):
            extras.append(f"gap_auto_closed={report.auto_closed_gap_count}")
        if not extras:
            extras.append("no_gap_needed")

        print(
            f"[Force close DXF via dxf_parser.py] {dxf_file.name} | "
            f"lines={report.line_count} | "
            + " | ".join(extras)
        )
        for warning in getattr(report, "warnings", []):
            print(f"  - {warning}")

    print(
        "[Force close DXF summary] "
        f"files={len(dxf_files)} | "
        f"path_auto_closed={total_path_auto_closed} | "
        f"gap_auto_closed={total_gap_auto_closed} | "
        f"warnings={total_warnings}"
    )
    return closed_dxf_dir



def _run_overlay_and_evaluation(
    *,
    args: argparse.Namespace,
    eval_dxf_dir: Path,
    original_dxf_dir: Path,
    gcode_dir: Path,
    param_csv: Path,
    timing_csv: Path,
    output_dir: Path,
    overlay_out_dir: Path,
) -> None:
    print("[Config]")
    print(f"  original_dxf_dir = {original_dxf_dir}")
    print(f"  eval_dxf_dir     = {eval_dxf_dir}")
    print(f"  gcode_dir        = {gcode_dir}")
    print(f"  param_csv        = {param_csv}")
    print(f"  timing_csv       = {timing_csv}")
    print(f"  output_dir       = {output_dir}")
    print(f"  overlay_out_dir  = {overlay_out_dir}")
    print(f"  force_close_dxf  = {args.force_close_dxf}")
    print(f"  close_tol_mm     = {args.close_tol_mm}")
    print(f"  flatten_distance = {args.flatten_distance}")
    print(f"  min_curve_segments = {args.min_curve_segments}")
    print(f"  tol_mm           = {args.tol_mm}")
    print(f"  path_width_mm    = {args.path_width_mm if args.path_width_mm is not None else 'AUTO'}")

    if not args.skip_overlay:
        print("\n[Step 1] Generate overlay PNG files")
        run_overlay_batch(
            dxf_dir=eval_dxf_dir,
            gcode_dir=gcode_dir,
            out_dir=overlay_out_dir,
        )
    else:
        print("\n[Step 1] Skip overlay PNG generation")

    print("\n[Step 2] Run evaluation")
    metrics_csv, summary_json = evaluate_all(
        dxf_dir=eval_dxf_dir,
        gcode_dir=gcode_dir,
        param_csv=param_csv,
        timing_csv=timing_csv,
        output_dir=output_dir,
        tol_mm=args.tol_mm,
        path_width_mm=args.path_width_mm,
    )
    print(f"[Done] Metrics CSV:   {metrics_csv}")
    print(f"[Done] Summary JSON:  {summary_json}")
    print(f"[Done] Overlay PNGs:  {overlay_out_dir}")



def main() -> None:
    args = build_argparser().parse_args()
    dxf_dir = _resolve_input_path(args.dxf_dir)
    gcode_dir = _resolve_input_path(args.gcode_dir)
    param_csv = _resolve_input_path(args.param_csv)
    timing_csv = _resolve_input_path(args.timing_csv)
    output_dir = _resolve_input_path(args.output_dir)
    overlay_out_dir = _resolve_input_path(args.overlay_out_dir)

    if overlay_out_dir is None:
        overlay_out_dir = output_dir / DEFAULT_OVERLAY_DIRNAME

    _validate_inputs(dxf_dir, gcode_dir, param_csv, timing_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_out_dir.mkdir(parents=True, exist_ok=True)

    if args.force_close_dxf:
        with tempfile.TemporaryDirectory(prefix="force_closed_dxf_parser_") as temp_dir:
            closed_dxf_dir = _prepare_force_closed_dxf_dir(
                dxf_dir=dxf_dir,
                temp_root=Path(temp_dir),
                args=args,
            )
            _run_overlay_and_evaluation(
                args=args,
                eval_dxf_dir=closed_dxf_dir,
                original_dxf_dir=dxf_dir,
                gcode_dir=gcode_dir,
                param_csv=param_csv,
                timing_csv=timing_csv,
                output_dir=output_dir,
                overlay_out_dir=overlay_out_dir,
            )
    else:
        _run_overlay_and_evaluation(
            args=args,
            eval_dxf_dir=dxf_dir,
            original_dxf_dir=dxf_dir,
            gcode_dir=gcode_dir,
            param_csv=param_csv,
            timing_csv=timing_csv,
            output_dir=output_dir,
            overlay_out_dir=overlay_out_dir,
        )


if __name__ == "__main__":
    main()
