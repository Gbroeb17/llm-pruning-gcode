from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path(
            "/workspace/llm-pruning-gcode/streamline_full_ep3_seqlen8192_cosine1.0_dxf1000_eval/metrics.csv"
        ),
        help="Path to metrics.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/workspace/llm-pruning-gcode/results/streamline/seqlen8192_cosine1.0/eval_1000/analysis"
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of worst samples to save",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # 1. Load metrics
    # ============================================================
    df = pd.read_csv(args.metrics)

    print(f"Loaded {len(df)} samples")
    print()

    # ============================================================
    # 2. IoU statistics
    # ============================================================
    iou = df["iou"]

    print("========== IoU Statistics ==========")
    print(f"Count  : {len(iou)}")
    print(f"Mean   : {iou.mean():.6f}")
    print(f"Median : {iou.median():.6f}")
    print(f"Std    : {iou.std():.6f}")
    print(f"Min    : {iou.min():.6f}")
    print(f"Max    : {iou.max():.6f}")
    print()

    print("IoU percentiles:")
    for q in [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
        print(f"P{int(q * 100):02d}: {iou.quantile(q):.6f}")
    print()

    # ============================================================
    # 3. IoU threshold analysis
    # ============================================================
    print("========== IoU Thresholds ==========")

    thresholds = [0.1, 0.5, 0.8, 0.9, 0.95, 0.99]

    for threshold in thresholds:
        count = (iou < threshold).sum()
        percentage = count / len(df) * 100

        print(
            f"IoU < {threshold:>4}: "
            f"{count:4d} / {len(df)} "
            f"({percentage:6.2f}%)"
        )

    print()

    # ============================================================
    # 4. IoU interval distribution
    # ============================================================
    bins = [
        -float("inf"),
        0.1,
        0.5,
        0.8,
        0.9,
        0.95,
        0.99,
        1.000001,
    ]

    labels = [
        "<0.1",
        "0.1-0.5",
        "0.5-0.8",
        "0.8-0.9",
        "0.9-0.95",
        "0.95-0.99",
        ">=0.99",
    ]

    df["iou_range"] = pd.cut(
        df["iou"],
        bins=bins,
        labels=labels,
        right=False,
    )

    distribution = (
        df["iou_range"]
        .value_counts(sort=False)
        .rename("count")
        .to_frame()
    )

    distribution["percentage"] = (
        distribution["count"] / len(df) * 100
    )

    print("========== IoU Distribution ==========")
    print(distribution)
    print()

    distribution.to_csv(
        output_dir / "iou_distribution.csv"
    )

    # ============================================================
    # 5. Worst IoU samples
    # ============================================================
    worst_iou = (
        df.sort_values("iou")
        .head(args.top_k)
    )

    print(f"========== Worst {args.top_k} IoU Samples ==========")

    print(
        worst_iou[
            [
                "filename",
                "iou",
                "coordinate_mae",
                "group_syntax_success_rate",
            ]
        ].to_string(index=False)
    )

    worst_iou.to_csv(
        output_dir / "worst_iou_samples.csv",
        index=False,
    )

    print()

    # ============================================================
    # 6. Coordinate MAE statistics
    # ============================================================
    coord = df["coordinate_mae"]

    print("========== Coordinate MAE ==========")
    print(f"Mean   : {coord.mean():.8f}")
    print(f"Median : {coord.median():.8f}")
    print(f"Std    : {coord.std():.8f}")
    print(f"Min    : {coord.min():.8f}")
    print(f"Max    : {coord.max():.8f}")
    print()

    worst_coord = (
        df.sort_values(
            "coordinate_mae",
            ascending=False,
        )
        .head(args.top_k)
    )

    print(
        f"========== Worst {args.top_k} "
        "Coordinate MAE Samples =========="
    )

    print(
        worst_coord[
            [
                "filename",
                "iou",
                "coordinate_mae",
            ]
        ].to_string(index=False)
    )

    worst_coord.to_csv(
        output_dir / "worst_coordinate_mae_samples.csv",
        index=False,
    )

    # ============================================================
    # 7. Low-IoU samples
    # ============================================================
    low_iou = df[df["iou"] < 0.5].sort_values("iou")

    low_iou.to_csv(
        output_dir / "low_iou_samples.csv",
        index=False,
    )

    # ============================================================
    # 8. Summary
    # ============================================================
    summary = {
        "num_samples": len(df),
        "iou_mean": iou.mean(),
        "iou_median": iou.median(),
        "iou_std": iou.std(),
        "iou_p10": iou.quantile(0.10),
        "iou_p25": iou.quantile(0.25),
        "iou_p50": iou.quantile(0.50),
        "iou_p75": iou.quantile(0.75),
        "iou_p90": iou.quantile(0.90),
        "iou_below_0.5": int((iou < 0.5).sum()),
        "iou_below_0.8": int((iou < 0.8).sum()),
        "iou_above_0.9": int((iou >= 0.9).sum()),
        "iou_above_0.99": int((iou >= 0.99).sum()),
        "coordinate_mae_mean": coord.mean(),
        "coordinate_mae_median": coord.median(),
        "coordinate_mae_max": coord.max(),
    }

    pd.DataFrame(
        summary.items(),
        columns=["metric", "value"],
    ).to_csv(
        output_dir / "analysis_summary.csv",
        index=False,
    )

    print()
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()