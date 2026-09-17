from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_detection_rate_vs_occlusion(df: pd.DataFrame, output_dir: Path) -> None:
    grouped = (
        df.groupby(["model", "occlusion_level"], as_index=False)["detected"]
        .mean()
        .assign(detection_rate=lambda x: x["detected"] * 100.0)
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for model, group in grouped.groupby("model"):
        group = group.sort_values("occlusion_level")
        ax.plot(
            group["occlusion_level"],
            group["detection_rate"],
            marker="o",
            label=model,
        )

    ax.set_title("Detection rate vs occlusion level")
    ax.set_xlabel("Occlusion level (%)")
    ax.set_ylabel("Detection rate (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save(fig, output_dir / "detection_rate_vs_occlusion.png")


def plot_target_confidence_vs_occlusion(df: pd.DataFrame, output_dir: Path) -> None:
    grouped = (
        df.groupby(["model", "occlusion_level"], as_index=False)["target_confidence"]
        .mean()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for model, group in grouped.groupby("model"):
        group = group.sort_values("occlusion_level")
        ax.plot(
            group["occlusion_level"],
            group["target_confidence"],
            marker="o",
            label=model,
        )

    ax.set_title("Mean target confidence vs occlusion level")
    ax.set_xlabel("Occlusion level (%)")
    ax.set_ylabel("Mean target confidence")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save(fig, output_dir / "target_confidence_vs_occlusion.png")


def plot_iou_vs_occlusion(df: pd.DataFrame, output_dir: Path) -> None:
    detected = df[df["detected"] == 1].copy()
    if detected.empty:
        return

    grouped = detected.groupby(
        ["model", "occlusion_level"], as_index=False
    )["target_iou"].mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    for model, group in grouped.groupby("model"):
        group = group.sort_values("occlusion_level")
        ax.plot(
            group["occlusion_level"],
            group["target_iou"],
            marker="o",
            label=model,
        )

    ax.set_title("Mean IoU of successful detections vs occlusion level")
    ax.set_xlabel("Occlusion level (%)")
    ax.set_ylabel("Mean IoU")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save(fig, output_dir / "iou_vs_occlusion.png")


def plot_detection_rate_by_class(df: pd.DataFrame, output_dir: Path) -> None:
    # Exclude 0% baseline so the chart describes robustness under occlusion.
    occluded = df[df["occlusion_level"] > 0].copy()
    if occluded.empty:
        return

    grouped = (
        occluded.groupby(["gt_class", "model"], as_index=False)["detected"]
        .mean()
        .assign(detection_rate=lambda x: x["detected"] * 100.0)
    )
    pivot = grouped.pivot(index="gt_class", columns="model", values="detection_rate")

    ax = pivot.plot(kind="bar", figsize=(9, 5))
    fig = ax.get_figure()
    ax.set_title("Mean detection rate by object class under occlusion")
    ax.set_xlabel("Object class")
    ax.set_ylabel("Detection rate (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=0)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(title="Model")
    _save(fig, output_dir / "detection_rate_by_class.png")


def plot_detection_rate_by_position(df: pd.DataFrame, output_dir: Path) -> None:
    occluded = df[df["occlusion_level"] > 0].copy()
    if occluded.empty:
        return

    grouped = (
        occluded.groupby(["occlusion_position", "model"], as_index=False)["detected"]
        .mean()
        .assign(detection_rate=lambda x: x["detected"] * 100.0)
    )
    pivot = grouped.pivot(
        index="occlusion_position", columns="model", values="detection_rate"
    )

    preferred_order = ["top", "bottom", "left", "right", "center", "centre"]
    existing = [x for x in preferred_order if x in pivot.index]
    remaining = [x for x in pivot.index if x not in existing]
    pivot = pivot.reindex(existing + remaining)

    ax = pivot.plot(kind="bar", figsize=(9, 5))
    fig = ax.get_figure()
    ax.set_title("Mean detection rate by occlusion position")
    ax.set_xlabel("Occlusion position")
    ax.set_ylabel("Detection rate (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=0)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(title="Model")
    _save(fig, output_dir / "detection_rate_by_position.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate plots from evaluate.py per-image output."
    )
    parser.add_argument(
        "--input",
        default="results/evaluation/per_image.csv",
        help="Path to per_image.csv produced by evaluate.py",
    )
    parser.add_argument(
        "--output-dir",
        default="results/evaluation/plots",
        help="Directory for PNG plots.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    required = {
        "model",
        "gt_class",
        "occlusion_level",
        "occlusion_position",
        "detected",
        "target_confidence",
        "target_iou",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {input_path}: {sorted(missing)}")

    plot_detection_rate_vs_occlusion(df, output_dir)
    plot_target_confidence_vs_occlusion(df, output_dir)
    plot_iou_vs_occlusion(df, output_dir)
    plot_detection_rate_by_class(df, output_dir)
    plot_detection_rate_by_position(df, output_dir)

    print("Done generating plots.")


if __name__ == "__main__":
    main()
