"""Scoring functions for Affordance Grounding (AFF) and Trajectory Planning (TRAJ)."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np


def resample_points(points: list[list[float]], target_length: int) -> np.ndarray:
    if len(points) == 0:
        return np.array([])

    points_np = np.array(points, dtype=float)

    if len(points_np) == 1:
        return np.tile(points_np, (target_length, 1))

    if len(points_np) == target_length:
        return points_np

    original_indices = np.linspace(0, len(points_np) - 1, len(points_np))
    target_indices = np.linspace(0, len(points_np) - 1, target_length)

    resampled_x = np.interp(target_indices, original_indices, points_np[:, 0])
    resampled_y = np.interp(target_indices, original_indices, points_np[:, 1])

    return np.column_stack((resampled_x, resampled_y))


def calculate_dfd(points: list[list[float]], ground_truth: list[list[float]]) -> float:
    resampled_points = resample_points(points, len(ground_truth))

    seq1 = np.array(resampled_points, dtype=float)
    seq2 = np.array(ground_truth, dtype=float)
    m, n = len(seq1), len(seq2)

    def dist_func(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    c = np.zeros((m, n), dtype=float)
    c[0, 0] = dist_func(seq1[0], seq2[0])

    for i in range(1, m):
        c[i, 0] = max(c[i - 1, 0], dist_func(seq1[i], seq2[0]))

    for j in range(1, n):
        c[0, j] = max(c[0, j - 1], dist_func(seq1[0], seq2[j]))

    for i in range(1, m):
        for j in range(1, n):
            c[i, j] = max(
                min(c[i - 1, j], c[i, j - 1], c[i - 1, j - 1]),
                dist_func(seq1[i], seq2[j]),
            )

    return float(c[m - 1, n - 1])


def create_predicted_box_original_size(
    predict_point: dict[str, Any], ground_truth_box: dict[str, Any]
) -> list[float]:
    x = float(predict_point["x"])
    y = float(predict_point["y"])
    x1_gt = float(ground_truth_box["x1"])
    y1_gt = float(ground_truth_box["y1"])
    x2_gt = float(ground_truth_box["x2"])
    y2_gt = float(ground_truth_box["y2"])

    gt_width = x2_gt - x1_gt
    gt_height = y2_gt - y1_gt
    half_width = gt_width / 2.0
    half_height = gt_height / 2.0

    return [x - half_width, y - half_height, x + half_width, y + half_height]


def calculate_iou(box1: list[float], box2: list[float]) -> float:
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height

    area1 = max(0.0, x2_1 - x1_1) * max(0.0, y2_1 - y1_1)
    area2 = max(0.0, x2_2 - x1_2) * max(0.0, y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def evaluate_affordance(data: list[dict[str, Any]]) -> tuple[float, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(data, start=1):
        gt_boxes = item.get("answers", [])
        model_answer = item.get("model_answer")
        score = 0.0
        pred_box: list[float] | None = None
        gt_box: list[float] | None = None
        matched_gt_index: int | None = None

        if (
            isinstance(gt_boxes, list)
            and gt_boxes
            and isinstance(model_answer, dict)
            and "x" in model_answer
            and "y" in model_answer
        ):
            # Each box is an alternative valid target, not a required extra answer.
            # Construct the prediction box using each candidate's own dimensions.
            for gt_index, gt in enumerate(gt_boxes):
                candidate_gt_box = [
                    float(gt["x1"]),
                    float(gt["y1"]),
                    float(gt["x2"]),
                    float(gt["y2"]),
                ]
                candidate_pred_box = create_predicted_box_original_size(model_answer, gt)
                candidate_score = calculate_iou(candidate_pred_box, candidate_gt_box) * 100.0
                if matched_gt_index is None or candidate_score > score:
                    score = candidate_score
                    pred_box = candidate_pred_box
                    gt_box = candidate_gt_box
                    matched_gt_index = gt_index

        rows.append(
            {
                "index": idx,
                "question_id": item.get("question_id"),
                "filename": item.get("filename"),
                "question": item.get("question"),
                "score": score,
                "answer": json.dumps(item.get("answer"), ensure_ascii=False),
                "model_answer": json.dumps(model_answer, ensure_ascii=False),
                "answers": json.dumps(gt_boxes, ensure_ascii=False),
                "pred_box": json.dumps(pred_box, ensure_ascii=False),
                "gt_box": json.dumps(gt_box, ensure_ascii=False),
                "matched_gt_index": matched_gt_index,
            }
        )

    average = float(sum(row["score"] for row in rows) / len(rows)) if rows else 0.0
    return average, rows


def normalize_trajectory_points(
    points: list[dict[str, Any]], width: float, height: float
) -> list[list[float]]:
    normalized: list[list[float]] = []
    for point in points:
        x = float(point["x"]) / width
        y = float(point["y"]) / height
        normalized.append([x, y])
    return normalized


def evaluate_trajectory(data: list[dict[str, Any]]) -> tuple[float, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(data, start=1):
        label = item.get("question_id") or item.get("filename") or str(idx)
        image_size = item.get("image_size")
        try:
            if not isinstance(image_size, dict):
                raise ValueError("missing image_size")
            width = float(image_size["width"])
            height = float(image_size["height"])
            if not (math.isfinite(width) and math.isfinite(height) and width > 0 and height > 0):
                raise ValueError("nonpositive or nonfinite dimensions")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{label}: image_size must contain positive finite width and height") from exc

        def valid_points(points: Any) -> bool:
            if not isinstance(points, list) or not points:
                return False
            try:
                return all(
                    isinstance(point, dict)
                    and not isinstance(point.get("x"), bool)
                    and not isinstance(point.get("y"), bool)
                    and math.isfinite(float(point["x"]))
                    and math.isfinite(float(point["y"]))
                    and 0 <= float(point["x"]) <= width
                    and 0 <= float(point["y"]) <= height
                    for point in points
                )
            except (KeyError, TypeError, ValueError):
                return False

        gt_points = item.get("trajectory", [])
        pred_points = item.get("model_trajectory", [])
        if not valid_points(gt_points):
            raise ValueError(f"{label}: ground-truth trajectory must contain finite in-image points")
        valid = valid_points(pred_points)
        # Maximum distance within the normalized unit square; retain invalid rows
        # in the denominator rather than rewarding them with DFD = 0.
        score = math.sqrt(2.0)
        if valid:
            norm_gt = normalize_trajectory_points(gt_points, width, height)
            norm_pred = normalize_trajectory_points(pred_points, width, height)
            score = calculate_dfd(norm_pred, norm_gt)

        rows.append(
            {
                "index": idx,
                "question_id": item.get("question_id"),
                "filename": item.get("filename"),
                "question": item.get("question"),
                "score": score,
                "valid_prediction": valid,
                "scoring_note": "computed_dfd" if valid else "invalid_prediction_max_dfd_penalty",
                "trajectory": json.dumps(gt_points, ensure_ascii=False),
                "model_trajectory": json.dumps(pred_points, ensure_ascii=False),
            }
        )

    average = float(sum(row["score"] for row in rows) / len(rows)) if rows else 0.0
    return average, rows
