#!/usr/bin/env python3
"""Build individual BEAKER model score summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


TAU_D = 0.40

LVPR_TASK = "Long-Video_Experimental_Procedure_Reasoning"


# ============================================================
# 1. Task score files
# ============================================================

TASK_FILES = [
    ("equipment_and_reagent_identification",
     "equipment_and_reagent_identification_scores.csv"),

    ("experimental_apparatus_setup",
     "experimental_apparatus_setup_scores.csv"),

    ("experimental_operations",
     "experimental_operations_scores.csv"),

    ("experimental_safety_and_standardization",
     "experimental_safety_and_standardization_scores.csv"),

    ("state_recognition",
     "state_recognition_scores.csv"),

    ("affordance",
     "affordance_scores.csv"),

    ("trajectory_dfd",
     "trajectory_scores.csv"),

    ("trajectory_score",
     "trajectory_scores.csv"),

    ("short_Temporal_Sequence_Reasoning",
     "short_Temporal_Sequence_Reasoning_scores.csv"),

    ("Short-Video_Operation_Reasoning",
     "Short-Video_Operation_Reasoning_scores.csv"),

    ("Mid-Video_Task-Step_Reasoning",
     "Mid-Video_Task-Step_Reasoning_scores.csv"),

    ("mid_Temporal_Sequence_Reasoning",
     "mid_Temporal_Sequence_Reasoning_scores.csv"),

    ("long_Temporal_Sequence_Reasoning",
     "long_Temporal_Sequence_Reasoning_scores.csv"),

    (LVPR_TASK,
     f"{LVPR_TASK}_scores.csv"),
]

EXPECTED_ROWS = {
    "equipment_and_reagent_identification_scores.csv": 60,
    "experimental_apparatus_setup_scores.csv": 60,
    "experimental_operations_scores.csv": 60,
    "experimental_safety_and_standardization_scores.csv": 60,
    "state_recognition_scores.csv": 60,
    "affordance_scores.csv": 150,
    "trajectory_scores.csv": 150,
    "short_Temporal_Sequence_Reasoning_scores.csv": 70,
    "Short-Video_Operation_Reasoning_scores.csv": 125,
    "Mid-Video_Task-Step_Reasoning_scores.csv": 75,
    "mid_Temporal_Sequence_Reasoning_scores.csv": 50,
    "long_Temporal_Sequence_Reasoning_scores.csv": 30,
    f"{LVPR_TASK}_scores.csv": 50,
}


# ============================================================
# 2. Read CSV metadata
# ============================================================

def read_header_metrics(path: Path) -> dict[str, str]:
    """Read summary metadata from a task CSV."""

    metrics: dict[str, str] = {}

    if not path.exists():
        return metrics

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)

        for row in reader:
            if not row:
                break

            if len(row) >= 2:
                metrics[row[0]] = row[1]

    return metrics


def read_detail_rows(path: Path) -> list[dict[str, str]]:
    """Read the question-level table below a task CSV's metadata block."""

    if not path.exists():
        return []

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)

        for row in reader:
            if not row:
                break
        else:
            return []

        return list(csv.DictReader(f))


def parse_float(value: str | None) -> float | None:
    """Convert a string to a finite float."""

    if value is None or value.strip() == "":
        return None

    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except ValueError:
        return None


def format_score(value: float | None) -> str:
    """Format score; missing values remain blank."""

    if value is None:
        return ""

    return f"{value:.6f}"


def calculate_expert_percentage(path: Path) -> float | None:
    """Calculate LVPR percentage from expert-entered question scores.

    Experts enter one score in the detail table for every question. ``score`` is
    preferred; ``blank_score`` accepts either one question score or a JSON list
    of per-blank scores such as ``[1, 0, 1]``. Per-blank scores are averaged
    within the question, then all question scores are averaged and multiplied by
    100. Scores must be in the inclusive range 0--1. Missing or invalid scores
    prevent summary calculation so an incomplete review cannot be mistaken for
    a complete evaluation.
    """

    rows = read_detail_rows(path)

    expected_rows = EXPECTED_ROWS[path.name]

    if len(rows) != expected_rows:
        print(
            f"[warn] {path}: expected {expected_rows} expert rows, found "
            f"{len(rows)}; LVPR and dependent totals left blank.",
            flush=True,
        )
        return None

    scores: list[float] = []

    for row_number, row in enumerate(rows, start=1):
        score_value = (row.get("score") or "").strip()
        blank_value = (row.get("blank_score") or "").strip()
        score = parse_float(score_value)

        if score is None and not score_value and blank_value:
            try:
                blank_scores = json.loads(blank_value)
            except json.JSONDecodeError:
                blank_scores = blank_value

            if isinstance(blank_scores, list):
                parsed_scores = [
                    parse_float(str(value))
                    for value in blank_scores
                ]

                try:
                    reference_answers = json.loads(
                        row.get("answer", "")
                    )
                except json.JSONDecodeError:
                    reference_answers = None

                expected_count = (
                    len(reference_answers)
                    if isinstance(reference_answers, list)
                    else None
                )

                if (
                    parsed_scores
                    and all(value is not None for value in parsed_scores)
                    and all(
                        0.0 <= value <= 1.0
                        for value in parsed_scores
                        if value is not None
                    )
                    and (
                        expected_count is None
                        or len(parsed_scores) == expected_count
                    )
                ):
                    score = sum(
                        value
                        for value in parsed_scores
                        if value is not None
                    ) / len(parsed_scores)
            else:
                score = parse_float(str(blank_scores))

        if score is None:
            print(
                f"[warn] {path}: expert score missing at detail row "
                f"{row_number}; LVPR and dependent totals left blank.",
                flush=True,
            )
            return None

        if not 0.0 <= score <= 1.0:
            print(
                f"[warn] {path}: expert score {score} at detail row "
                f"{row_number} is outside 0--1; LVPR and dependent totals "
                "left blank.",
                flush=True,
            )
            return None

        scores.append(score)

    return sum(scores) / len(scores) * 100.0


def has_complete_coverage(path: Path) -> bool:
    """Return whether an automatic task CSV contains every expected item."""

    expected_rows = EXPECTED_ROWS[path.name]
    actual_rows = len(read_detail_rows(path))

    if actual_rows == expected_rows:
        return True

    print(
        f"[warn] {path}: expected {expected_rows} result rows, found "
        f"{actual_rows}; task and dependent totals left blank.",
        flush=True,
    )
    return False


# ============================================================
# 3. Weighted score calculation
# ============================================================

def weighted_average(
    values: list[tuple[float | None, float]],
) -> float | None:
    """Calculate weighted average.

    Return None if any required score is missing.
    """

    if any(score is None for score, _ in values):
        return None

    total_weight = sum(weight for _, weight in values)

    if total_weight <= 0:
        return None

    return sum(
        score * weight
        for score, weight in values
        if score is not None
    ) / total_weight


def get_score(
    row: dict[str, str],
    name: str,
) -> float | None:
    """Get a numeric task score from the summary row."""

    return parse_float(row.get(name))


# ============================================================
# 4. Capability scores
# ============================================================

def calculate_capability_scores(
    row: dict[str, str],
) -> None:
    """Calculate four capability scores and Overall."""

    # Actionability: AFF + TRAJ
    actionability = weighted_average([
        (get_score(row, "affordance"), 1),
        (get_score(row, "trajectory_score"), 1),
    ])

    # Knowledge: ERI + ESS + SR
    knowledge = weighted_average([
        (get_score(row, "equipment_and_reagent_identification"), 1),
        (get_score(row, "experimental_safety_and_standardization"), 1),
        (get_score(row, "state_recognition"), 1),
    ])

    # Experimentation: EAS + EOP + SVOR + MVTS
    # Image-based tasks weight 2; video-based tasks weight 5.
    experimentation = weighted_average([
        (get_score(row, "experimental_apparatus_setup"), 2),
        (get_score(row, "experimental_operations"), 2),
        (get_score(row, "Short-Video_Operation_Reasoning"), 5),
        (get_score(row, "Mid-Video_Task-Step_Reasoning"), 5),
    ])

    # Temporal Sequence Reasoning: SV + MV + LV
    tsr = weighted_average([
        (get_score(row, "short_Temporal_Sequence_Reasoning"), 1),
        (get_score(row, "mid_Temporal_Sequence_Reasoning"), 1),
        (get_score(row, "long_Temporal_Sequence_Reasoning"), 1),
    ])

    # Long-Video Experimental Procedure Reasoning: expert-evaluated task
    lvpr = get_score(row, LVPR_TASK)

    # Reasoning: TSR + LVPR
    reasoning = weighted_average([
        (tsr, 1),
        (lvpr, 1),
    ])

    # Overall: weighted combination of four capabilities
    overall = weighted_average([
        (actionability, 0.25),
        (knowledge, 0.15),
        (experimentation, 0.35),
        (reasoning, 0.25),
    ])

    row["TSR"] = format_score(tsr)

    row["Actionability"] = format_score(actionability)
    row["Knowledge"] = format_score(knowledge)
    row["Experimentation"] = format_score(experimentation)
    row["Reasoning"] = format_score(reasoning)

    row["Overall"] = format_score(overall)


# ============================================================
# 5. Discover model directories
# ============================================================

def discover_model_dirs(result_root: Path) -> list[Path]:
    """Find model directories containing evaluation results."""

    model_dirs = []

    if not result_root.is_dir():
        return model_dirs

    for path in sorted(result_root.iterdir()):

        if not path.is_dir():
            continue

        model_dirs.append(path)

    return model_dirs


# ============================================================
# 6. Build individual model summaries
# ============================================================

def build_summary(
    result_root: Path,
) -> None:
    """Generate per-model and combined score summary CSV files."""

    fieldnames = (
        ["model"]
        + [name for name, _ in TASK_FILES]
        + [
            "TSR",
            "Actionability",
            "Knowledge",
            "Experimentation",
            "Reasoning",
            "Overall",
        ]
    )

    model_dirs = discover_model_dirs(result_root)
    summary_rows: list[dict[str, str]] = []
    coverage_cache: dict[Path, bool] = {}

    for model_dir in model_dirs:

        row: dict[str, str] = {
            "model": model_dir.name
        }

        # Read all task scores.
        for column_name, filename in TASK_FILES:

            score_path = model_dir / filename
            metrics = read_header_metrics(score_path)

            if (
                column_name != LVPR_TASK
                and score_path not in coverage_cache
            ):
                coverage_cache[score_path] = has_complete_coverage(score_path)

            if (
                column_name != LVPR_TASK
                and not coverage_cache[score_path]
            ):
                row[column_name] = ""
                continue

            if column_name == "trajectory_dfd":

                dfd = parse_float(
                    metrics.get("average_dfd")
                )

                row[column_name] = format_score(dfd)

            elif column_name == "trajectory_score":

                dfd = parse_float(
                    metrics.get("average_dfd")
                )

                score = (
                    100.0 * math.exp(-dfd / TAU_D)
                    if dfd is not None
                    else None
                )

                row[column_name] = format_score(score)

            elif column_name == LVPR_TASK:

                percentage = calculate_expert_percentage(
                    score_path
                )

                row[column_name] = format_score(percentage)

            else:

                percentage = parse_float(
                    metrics.get("percentage")
                )

                row[column_name] = format_score(percentage)

        # Calculate capability scores and Overall.
        calculate_capability_scores(row)
        summary_rows.append(row)

        # Each model has its own summary file.
        output_path = model_dir / "score_summary.csv"

        with output_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerow(row)

        print(
            f"[done] {model_dir.name} -> {output_path}",
            flush=True,
        )

    combined_path = result_root / "all_model_score_summary.csv"
    combined_path.parent.mkdir(parents=True, exist_ok=True)

    with combined_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(
        f"[summary] Generated {len(model_dirs)} model summaries -> "
        f"{combined_path}",
        flush=True,
    )


# ============================================================
# 7. Command-line arguments
# ============================================================

def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description="Build BEAKER per-model and combined score summaries."
    )

    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path(__file__).resolve().parent / "result",
    )

    return parser


# ============================================================
# 8. Main
# ============================================================

def main() -> int:

    args = build_parser().parse_args()

    build_summary(
        args.result_root.resolve()
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
