from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd

from metrics import (
    add_detection_retention,
    evaluate_image_record,
    robustness_o80,
    summarize_groups,
)


def load_result_file(path: Path) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Supported JSON formats:

    1) Preferred:
       {
         "model": "yolo26n",
         "images": [ ... ]
       }

    2) Alternative:
       {
         "model": "yolo26n",
         "results": [ ... ]
       }

    3) Plain list of per-image records:
       [ ... ]
       In this case the model name is derived from the filename.
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        model_name = path.stem
        records = data
    elif isinstance(data, dict):
        model_name = str(data.get("model", data.get("model_name", path.stem)))
        records = data.get("images", data.get("results"))
        if records is None:
            raise ValueError(
                f"{path}: expected top-level 'images' or 'results' list."
            )
    else:
        raise ValueError(f"{path}: unsupported JSON structure.")

    if not isinstance(records, list):
        raise ValueError(f"{path}: image/result records must be a list.")

    return model_name, records


def discover_json_files(inputs: List[str], input_dir: str | None) -> List[Path]:
    paths = [Path(p) for p in inputs]

    if input_dir:
        base = Path(input_dir)
        if not base.exists():
            raise FileNotFoundError(base)
        paths.extend(sorted(base.glob("*/results.json")))
        paths.extend(sorted(base.glob("*.json")))

    # Remove duplicates while preserving order.
    seen = set()
    unique: List[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)

    if not unique:
        raise ValueError("No detector result JSON files were provided/found.")

    return unique


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate common-format JSON outputs from multiple object detectors."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Detector result JSON files, e.g. results/rf_detr/results.json",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Optional results directory. Searches */results.json and *.json.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/evaluation",
        help="Directory for evaluation CSV/JSON outputs.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Minimum detection confidence. Default: 0.25",
    )
    parser.add_argument(
        "--matching",
        choices=["center", "iou"],
        default="center",
        help="Target association rule. Default: center",
    )
    parser.add_argument(
        "--min-iou",
        type=float,
        default=0.50,
        help="Minimum IoU when --matching iou is used. Default: 0.50",
    )
    args = parser.parse_args()

    json_files = discover_json_files(args.inputs, args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_image_rows: List[Dict[str, Any]] = []

    for path in json_files:
        model_name, records = load_result_file(path)
        print(f"Evaluating {model_name}: {len(records)} images from {path}")

        for record in records:
            row = evaluate_image_record(
                record,
                model_name=model_name,
                confidence_threshold=args.confidence,
                matching=args.matching,
                min_iou=args.min_iou,
            )
            per_image_rows.append(row)

    df = pd.DataFrame(per_image_rows)
    if df.empty:
        raise ValueError("No image records were evaluated.")

    # Keep occlusion numerically sortable and present integer levels cleanly.
    df["occlusion_level"] = pd.to_numeric(df["occlusion_level"], errors="raise")

    save_table(df, output_dir / "per_image.csv")

    overall = summarize_groups(df, ["model"])
    save_table(overall, output_dir / "summary_overall.csv")

    by_occlusion = summarize_groups(df, ["model", "occlusion_level"])
    by_occlusion = add_detection_retention(by_occlusion, baseline_keys=["model"])
    save_table(by_occlusion, output_dir / "summary_by_occlusion.csv")

    by_class = summarize_groups(df, ["model", "gt_class"])
    save_table(by_class, output_dir / "summary_by_class.csv")

    # 0%-occlusion images normally have position='none'; exclude those here.
    occluded_df = df[df["occlusion_level"] > 0].copy()
    by_position = summarize_groups(occluded_df, ["model", "occlusion_position"])
    save_table(by_position, output_dir / "summary_by_position.csv")

    by_class_occlusion = summarize_groups(
        df, ["model", "gt_class", "occlusion_level"]
    )
    by_class_occlusion = add_detection_retention(
        by_class_occlusion,
        baseline_keys=["model", "gt_class"],
    )
    save_table(by_class_occlusion, output_dir / "summary_by_class_occlusion.csv")

    by_position_occlusion = summarize_groups(
        occluded_df, ["model", "occlusion_position", "occlusion_level"]
    )
    save_table(
        by_position_occlusion,
        output_dir / "summary_by_position_occlusion.csv",
    )

    o80 = robustness_o80(by_occlusion)
    save_table(o80, output_dir / "robustness_o80.csv")

    summary_payload = {
        "settings": {
            "confidence_threshold": args.confidence,
            "matching": args.matching,
            "min_iou": args.min_iou,
        },
        "overall": overall.to_dict(orient="records"),
        "by_occlusion": by_occlusion.to_dict(orient="records"),
        "robustness_o80": o80.to_dict(orient="records"),
    }
    summary_json = output_dir / "summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)
    print(f"Saved: {summary_json}")

    print("\nDone.")
    print(f"Evaluated {len(df)} image/model records across {df['model'].nunique()} model(s).")


if __name__ == "__main__":
    main()
