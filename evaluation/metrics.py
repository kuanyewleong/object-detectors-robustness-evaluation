from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# Canonical COCO-style names used in this project.
CLASS_ALIASES = {
    "cup": "cup",
    "cups": "cup",
    "book": "book",
    "books": "book",
    "plant": "potted plant",
    "plants": "potted plant",
    "potted plant": "potted plant",
    "potted plants": "potted plant",
    "potted_plant": "potted plant",
    "potted_plants": "potted plant",
}


def canonical_class_name(name: Any) -> str:
    """Normalize class names so all detectors use the same labels."""
    if name is None:
        return ""
    value = str(name).strip().lower().replace("-", " ")
    value = " ".join(value.split())
    return CLASS_ALIASES.get(value, value)


def _as_bbox(bbox: Sequence[float]) -> Tuple[float, float, float, float]:
    if bbox is None or len(bbox) != 4:
        raise ValueError(f"Bounding box must be [x1, y1, x2, y2], got: {bbox}")
    x1, y1, x2, y2 = map(float, bbox)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bounding box coordinates: {bbox}")
    return x1, y1, x2, y2


def box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Intersection over Union for two XYXY bounding boxes."""
    ax1, ay1, ax2, ay2 = _as_bbox(box_a)
    bx1, by1, bx2, by2 = _as_bbox(box_b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - intersection

    return float(intersection / union) if union > 0 else 0.0


def box_center_inside(pred_box: Sequence[float], gt_box: Sequence[float]) -> bool:
    """Return True when the predicted box centre lies inside the GT box."""
    px1, py1, px2, py2 = _as_bbox(pred_box)
    gx1, gy1, gx2, gy2 = _as_bbox(gt_box)
    cx = (px1 + px2) / 2.0
    cy = (py1 + py2) / 2.0
    return gx1 <= cx <= gx2 and gy1 <= cy <= gy2


def _normalize_detection(det: Dict[str, Any]) -> Dict[str, Any]:
    class_name = det.get("class_name", det.get("label", det.get("class")))
    confidence = det.get("confidence", det.get("score", 0.0))
    bbox = det.get("bbox", det.get("xyxy"))

    if bbox is None:
        raise ValueError(f"Detection is missing bbox/xyxy: {det}")

    return {
        "class_name": canonical_class_name(class_name),
        "confidence": float(confidence),
        "bbox": list(map(float, bbox)),
    }


def _extract_gt(image_record: Dict[str, Any]) -> Dict[str, Any]:
    """Support either nested `gt` metadata or equivalent top-level fields."""
    gt = dict(image_record.get("gt", {}))

    def pick(key: str, *alternatives: str, default: Any = None) -> Any:
        if key in gt:
            return gt[key]
        for alt in alternatives:
            if alt in gt:
                return gt[alt]
        if key in image_record:
            return image_record[key]
        for alt in alternatives:
            if alt in image_record:
                return image_record[alt]
        return default

    return {
        "class_name": canonical_class_name(
            pick("class_name", "gt_class", "label", default="")
        ),
        "bbox": pick("bbox", "gt_bbox"),
        "occlusion_level": float(
            pick("occlusion_level", "occlusion", "occlusion_percent", default=0)
        ),
        "occlusion_position": str(
            pick("occlusion_position", "position", default="none")
        ).strip().lower(),
        "object_id": str(pick("object_id", "instance_id", default="")),
        "viewpoint": str(pick("viewpoint", "view", default="")),
        "background": str(pick("background", "background_type", default="")),
    }


def _is_associated(
    det_bbox: Sequence[float],
    gt_bbox: Sequence[float],
    matching: str,
    min_iou: float,
) -> bool:
    if matching == "center":
        return box_center_inside(det_bbox, gt_bbox)
    if matching == "iou":
        return box_iou(det_bbox, gt_bbox) >= min_iou
    raise ValueError("matching must be either 'center' or 'iou'")


def evaluate_image_record(
    image_record: Dict[str, Any],
    model_name: str,
    confidence_threshold: float = 0.25,
    matching: str = "center",
    min_iou: float = 0.50,
) -> Dict[str, Any]:
    """
    Evaluate one image containing one target object.

    Primary success criterion (default):
      1. predicted class matches GT class,
      2. confidence >= threshold,
      3. predicted box centre lies inside the GT box.

    IoU is recorded as a secondary localization metric.
    """
    gt = _extract_gt(image_record)
    gt_class = gt["class_name"]
    gt_bbox = gt["bbox"]

    if not gt_class:
        raise ValueError(f"Missing GT class in record: {image_record.get('image_id')}")
    if gt_bbox is None:
        raise ValueError(f"Missing GT bbox in record: {image_record.get('image_id')}")

    detections = [
        _normalize_detection(d)
        for d in image_record.get("detections", image_record.get("predictions", []))
    ]

    above_threshold = [
        d for d in detections if d["confidence"] >= confidence_threshold
    ]
    associated = [
        d
        for d in above_threshold
        if _is_associated(d["bbox"], gt_bbox, matching, min_iou)
    ]
    correct = [d for d in associated if d["class_name"] == gt_class]

    best_correct: Optional[Dict[str, Any]] = (
        max(correct, key=lambda d: d["confidence"]) if correct else None
    )
    best_any: Optional[Dict[str, Any]] = (
        max(associated, key=lambda d: d["confidence"]) if associated else None
    )

    detected = best_correct is not None
    wrong_class = (not detected) and (best_any is not None)
    missed = best_any is None

    selected = best_correct if detected else best_any
    selected_class = selected["class_name"] if selected else ""
    selected_conf = float(selected["confidence"]) if selected else 0.0
    selected_iou = box_iou(selected["bbox"], gt_bbox) if selected else 0.0

    # Target confidence is deliberately set to zero on a miss/wrong-class result.
    target_confidence = float(best_correct["confidence"]) if best_correct else 0.0
    target_iou = box_iou(best_correct["bbox"], gt_bbox) if best_correct else 0.0

    return {
        "model": model_name,
        "image_id": str(
            image_record.get("image_id", image_record.get("file_name", image_record.get("image_path", "")))
        ),
        "image_path": str(image_record.get("image_path", image_record.get("file_name", ""))),
        "gt_class": gt_class,
        "gt_bbox": list(map(float, gt_bbox)),
        "occlusion_level": gt["occlusion_level"],
        "occlusion_position": gt["occlusion_position"],
        "object_id": gt["object_id"],
        "viewpoint": gt["viewpoint"],
        "background": gt["background"],
        "detected": int(detected),
        "wrong_class": int(wrong_class),
        "missed": int(missed),
        "predicted_class": selected_class,
        "predicted_confidence": selected_conf,
        "target_confidence": target_confidence,
        "target_iou": target_iou,
        "selected_iou": selected_iou,
        "num_detections_above_threshold": len(above_threshold),
    }


def summarize_groups(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """Create a compact metric table for any requested grouping."""
    if df.empty:
        return pd.DataFrame()

    grouped = df.groupby(group_cols, dropna=False, sort=True)
    rows: List[Dict[str, Any]] = []

    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = dict(zip(group_cols, keys))
        detected_mask = group["detected"].astype(bool)

        row.update(
            {
                "n_images": int(len(group)),
                "n_detected": int(group["detected"].sum()),
                "detection_rate": float(group["detected"].mean() * 100.0),
                "wrong_class_rate": float(group["wrong_class"].mean() * 100.0),
                "miss_rate": float(group["missed"].mean() * 100.0),
                "mean_target_confidence": float(group["target_confidence"].mean()),
                "mean_confidence_when_detected": (
                    float(group.loc[detected_mask, "target_confidence"].mean())
                    if detected_mask.any()
                    else np.nan
                ),
                "mean_iou_when_detected": (
                    float(group.loc[detected_mask, "target_iou"].mean())
                    if detected_mask.any()
                    else np.nan
                ),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def add_detection_retention(
    summary: pd.DataFrame,
    baseline_keys: List[str],
    occlusion_col: str = "occlusion_level",
) -> pd.DataFrame:
    """
    Add detection retention relative to each group's 0%-occlusion baseline.

    retention = current detection rate / baseline detection rate * 100
    """
    if summary.empty:
        return summary.copy()

    out = summary.copy()
    baseline = out[out[occlusion_col] == 0][baseline_keys + ["detection_rate"]].copy()
    baseline = baseline.rename(columns={"detection_rate": "baseline_detection_rate"})

    out = out.merge(baseline, on=baseline_keys, how="left")
    out["detection_retention"] = np.where(
        out["baseline_detection_rate"] > 0,
        out["detection_rate"] / out["baseline_detection_rate"] * 100.0,
        np.nan,
    )
    return out


def robustness_o80(summary_by_occlusion: pd.DataFrame) -> pd.DataFrame:
    """
    Highest tested occlusion level where detection retention is >= 80%.
    Computed per model.
    """
    rows: List[Dict[str, Any]] = []
    if summary_by_occlusion.empty:
        return pd.DataFrame(columns=["model", "o80"])

    for model, group in summary_by_occlusion.groupby("model"):
        eligible = group[
            (group["occlusion_level"] > 0)
            & (group["detection_retention"].notna())
            & (group["detection_retention"] >= 80.0)
        ]
        o80 = float(eligible["occlusion_level"].max()) if not eligible.empty else np.nan
        rows.append({"model": model, "o80": o80})

    return pd.DataFrame(rows)
