#!/usr/bin/env python3
"""Evaluate all BEAKER answer types for every model under exam/answer.

Automatically evaluates:
    - 5 image MCQ categories
    - 5 video MCQ categories
    - Affordance
    - Trajectory

Generates an unscored expert evaluation template for:
    - Long-video fill-in-the-blank

Output:
    evaluation/result/<model>/*_scores.csv

Important:
    Existing expert evaluation CSV files are never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from affordance_trajectory_eval import evaluate_affordance, evaluate_trajectory
from image_mcq_eval import CATEGORIES, build_rows, load_json as load_image_json


# ============================================================
# 1. Task configuration
# ============================================================

VIDEO_MCQ_FILES = [
    "short_Temporal_Sequence_Reasoning_answer.json",
    "Short-Video_Operation_Reasoning_answer.json",
    "Mid-Video_Task-Step_Reasoning_answer.json",
    "mid_Temporal_Sequence_Reasoning_answer.json",
    "long_Temporal_Sequence_Reasoning_answer.json",
]

CHOICES = {"A", "B", "C", "D"}

# Expert-evaluated fill-in-the-blank task
FILL_BLANK_TASK = "Long-Video_Experimental_Procedure_Reasoning"

FILL_BLANK_INPUT = f"{FILL_BLANK_TASK}_answer.json"
FILL_BLANK_OUTPUT = f"{FILL_BLANK_TASK}_scores.csv"


# ============================================================
# 2. Common utilities
# ============================================================

def write_rows_csv(
    path: Path,
    header_rows: list[tuple[str, str]],
    rows: list[dict[str, Any]],
) -> None:
    """Write a CSV with summary metadata and question-level results."""

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys()) if rows else ["index", "score"]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)

        for key, value in header_rows:
            writer.writerow([key, value])

        writer.writerow([])
        writer.writerow(fieldnames)

        dict_writer = csv.DictWriter(f, fieldnames=fieldnames)

        for row in rows:
            dict_writer.writerow(row)


def load_json(path: Path) -> list[dict[str, Any]]:
    """Load JSON annotation file."""

    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON array.")

    return data


def normalize_choice(value: Any) -> str:
    """Normalize MCQ answer to A, B, C, or D."""

    if value is None:
        return ""

    if isinstance(value, list):
        value = " ".join(str(item) for item in value)

    if isinstance(value, dict):
        value = json.dumps(value, ensure_ascii=False)

    raw = str(value).strip().upper()

    if raw in CHOICES:
        return raw

    match = re.search(r"\b([ABCD])\b", raw)

    return match.group(1) if match else raw


# ============================================================
# 3. Video MCQ evaluation
# ============================================================

def evaluate_video_mcq_file(
    path: Path,
    task_name: str,
) -> tuple[float, float, list[dict[str, Any]]]:
    """Evaluate a video MCQ JSON file."""

    if not path.exists():
        return 0.0, 0.0, []

    rows: list[dict[str, Any]] = []
    score_sum = 0.0

    data = load_json(path)

    for index, item in enumerate(data, start=1):

        answer = normalize_choice(item.get("answer"))
        model_answer = normalize_choice(item.get("model_answer"))

        score = (
            1.0
            if answer in CHOICES and answer == model_answer
            else 0.0
        )

        score_sum += score

        rows.append(
            {
                "task": task_name,
                "index": index,
                "filename": item.get("filename"),
                "question": item.get("question"),
                "answer": answer,
                "model_answer": model_answer,
                "score": f"{score:.6f}",
                "judge": "exact_choice",
            }
        )

    return score_sum, float(len(data)), rows


# ============================================================
# 4. Image MCQ evaluation
# ============================================================

def write_image_category_csv(
    *,
    model_dir: Path,
    output_dir: Path,
    category_name: str,
    config: dict[str, str],
) -> None:
    """Evaluate simple and complex image MCQs."""

    simple_path = model_dir / config["simple"]
    complex_path = model_dir / config["complex"]

    if not simple_path.exists() and not complex_path.exists():
        return

    simple_data = (
        load_image_json(simple_path)
        if simple_path.exists()
        else []
    )

    complex_data = (
        load_image_json(complex_path)
        if complex_path.exists()
        else []
    )

    simple_score, simple_rows = build_rows(
        simple_data,
        "simple",
    )

    complex_score, complex_rows = build_rows(
        complex_data,
        "complex",
    )

    rows = simple_rows + complex_rows

    total_score = simple_score + complex_score
    total_points = len(rows)

    percentage = (
        total_score / total_points * 100.0
        if total_points
        else 0.0
    )

    output_path = output_dir / config["output"]

    write_rows_csv(
        output_path,
        [
            ("task", category_name),
            ("raw_score", f"{total_score:.6f}"),
            ("total_points", f"{total_points:.6f}"),
            ("percentage", f"{percentage:.6f}"),
            (
                "simple_score",
                f"{simple_score}/{len(simple_rows)}",
            ),
            (
                "complex_score",
                f"{complex_score}/{len(complex_rows)}",
            ),
        ],
        rows,
    )


# ============================================================
# 5. Affordance evaluation
# ============================================================

def write_affordance_csv(
    model_dir: Path,
    output_dir: Path,
) -> None:
    """Evaluate affordance localization."""

    path = model_dir / "affordance_answer.json"

    if not path.exists():
        return

    data = load_json(path)

    average, rows = evaluate_affordance(data)

    raw_score = sum(
        float(row["score"])
        for row in rows
    )

    total_points = len(rows) * 100.0

    write_rows_csv(
        output_dir / "affordance_scores.csv",
        [
            ("task", "affordance"),
            ("raw_score", f"{raw_score:.6f}"),
            ("total_points", f"{total_points:.6f}"),
            ("percentage", f"{average:.6f}"),
            (
                "metric",
                "Maximum IoU percent over all valid GT boxes; "
                "higher is better",
            ),
        ],
        rows,
    )


# ============================================================
# 6. Trajectory evaluation
# ============================================================

def write_trajectory_csv(
    model_dir: Path,
    output_dir: Path,
) -> None:
    """Evaluate trajectory planning using DFD."""

    path = model_dir / "trajectory_answer.json"

    if not path.exists():
        return

    data = load_json(path)

    average, rows = evaluate_trajectory(data)

    raw_score = sum(
        float(row["score"])
        for row in rows
    )

    write_rows_csv(
        output_dir / "trajectory_scores.csv",
        [
            ("task", "trajectory"),
            ("raw_score", f"{raw_score:.6f}"),
            ("total_items", f"{len(rows):.6f}"),
            ("average_dfd", f"{average:.6f}"),
            (
                "valid_predictions",
                str(
                    sum(
                        row["valid_prediction"]
                        for row in rows
                    )
                ),
            ),
            (
                "invalid_predictions",
                str(
                    sum(
                        not row["valid_prediction"]
                        for row in rows
                    )
                ),
            ),
            (
                "invalid_prediction_policy",
                "DFD=sqrt(2); included in average",
            ),
            (
                "metric",
                "discrete Frechet distance; lower is better",
            ),
        ],
        rows,
    )


# ============================================================
# 7. Batch video MCQ evaluation
# ============================================================

def write_video_csvs(
    *,
    model_dir: Path,
    output_dir: Path,
) -> None:
    """Evaluate all video MCQ categories."""

    for filename in VIDEO_MCQ_FILES:

        task_name = filename.removesuffix(
            "_answer.json"
        )

        score, total, rows = evaluate_video_mcq_file(
            model_dir / filename,
            task_name,
        )

        if total == 0 and not rows:
            continue

        percentage = (
            score / total * 100.0
            if total
            else 0.0
        )

        write_rows_csv(
            output_dir / f"{task_name}_scores.csv",
            [
                ("task", task_name),
                ("raw_score", f"{score:.6f}"),
                ("total_points", f"{total:.6f}"),
                ("percentage", f"{percentage:.6f}"),
            ],
            rows,
        )


# ============================================================
# 8. Fill-in-the-blank expert evaluation template
# ============================================================

def serialize_fill_blank_answer(value: Any) -> str:
    """Preserve reference answers and model responses."""

    if value is None:
        return ""

    return json.dumps(
        value,
        ensure_ascii=False,
    )


def write_fill_blank_csv(
    model_dir: Path,
    output_dir: Path,
) -> None:
    """Generate an unscored expert evaluation CSV.

    This function:
        - Reads fill-in-the-blank answer JSON.
        - Preserves questions and model responses.
        - Leaves all scoring fields empty.
        - Never overwrites existing expert evaluation CSVs.
    """

    input_path = model_dir / FILL_BLANK_INPUT
    output_path = output_dir / FILL_BLANK_OUTPUT

    if not input_path.exists():
        print(
            f"[skip] {model_dir.name}: "
            f"{FILL_BLANK_INPUT} not found",
            flush=True,
        )
        return

    # Protect existing expert evaluation results.
    if output_path.exists():
        print(
            f"[keep] {model_dir.name}: "
            f"existing expert CSV preserved",
            flush=True,
        )
        return

    data = load_json(input_path)

    rows: list[dict[str, Any]] = []

    for index, item in enumerate(data, start=1):

        answer = item.get("answer")
        model_answer = item.get("model_answer")

        # Number of blanks in the reference answer.
        if isinstance(answer, list):
            blank_count = len(answer)
        elif answer is None:
            blank_count = 0
        else:
            blank_count = 1

        # Equal weight for each blank.
        blank_weight = (
            1.0 / blank_count
            if blank_count > 0
            else None
        )

        rows.append(
            {
                "task": FILL_BLANK_TASK,
                "index": index,
                "filename": item.get(
                    "filename",
                    "",
                ),
                "question": item.get(
                    "question",
                    "",
                ),
                "answer": serialize_fill_blank_answer(
                    answer
                ),
                "model_answer": serialize_fill_blank_answer(
                    model_answer
                ),
                "blank_index": "all",
                "blank_weight": (
                    f"{blank_weight:.6f}"
                    if blank_weight is not None
                    else ""
                ),

                # All expert scoring fields remain blank.
                "blank_score": "",
                "score": "",
                "judge": "",
            }
        )

    write_rows_csv(
        output_path,
        [
            ("task", FILL_BLANK_TASK),
            ("raw_score", ""),
            ("total_points", str(len(rows))),
            ("percentage", ""),
            ("judge", ""),
        ],
        rows,
    )

    print(
        f"[expert] {model_dir.name}: "
        f"{len(rows)} questions -> {output_path}",
        flush=True,
    )


# ============================================================
# 9. Model discovery and summary
# ============================================================

def discover_models(
    answer_root: Path,
    requested: list[str] | None,
) -> list[str]:

    if requested:
        return requested

    return sorted(
        path.name
        for path in answer_root.iterdir()
        if path.is_dir()
    )


def count_csv_files(path: Path) -> int:
    return len(
        list(path.glob("*.csv"))
    )


def warn_api_failures(model_dir: Path) -> None:
    """Report questions whose API request failed; they are scored as wrong."""

    for path in sorted(model_dir.glob("*_answer.json")):
        failed = sum(
            1 for item in load_json(path) if "api_error" in item
        )
        if failed:
            print(
                f"[warn] {model_dir.name}: {failed} questions in {path.name} "
                "have no answer because the API request failed and are "
                "scored as wrong. Rerun inference with --skip-existing to "
                "retry them.",
                flush=True,
            )


# ============================================================
# 10. Command-line arguments
# ============================================================

def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate all BEAKER benchmark answer files "
            "and generate expert scoring templates."
        )
    )

    parser.add_argument(
        "--answer-root",
        type=Path,
        default=(
            Path(__file__).resolve().parent.parent
            / "exam"
            / "answer"
        ),
    )

    parser.add_argument(
        "--result-root",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "result"
        ),
    )

    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
    )

    return parser


# ============================================================
# 11. Main
# ============================================================

def main() -> int:

    args = build_parser().parse_args()

    answer_root = args.answer_root.resolve()
    result_root = args.result_root.resolve()

    if not answer_root.is_dir():
        raise FileNotFoundError(
            f"Answer directory not found: {answer_root}"
        )

    result_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_names = discover_models(
        answer_root,
        args.models,
    )

    for model_name in model_names:

        model_dir = answer_root / model_name

        if not model_dir.is_dir():
            print(
                f"[warn] missing_model_dir {model_dir}",
                flush=True,
            )
            continue

        output_dir = result_root / model_name

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            f"[model] {model_name} -> {output_dir}",
            flush=True,
        )

        warn_api_failures(model_dir)

        # 1. Image MCQ: 5 categories
        for category_name, config in CATEGORIES.items():

            write_image_category_csv(
                model_dir=model_dir,
                output_dir=output_dir,
                category_name=category_name,
                config=config,
            )

        # 2. Affordance
        write_affordance_csv(
            model_dir,
            output_dir,
        )

        # 3. Trajectory
        write_trajectory_csv(
            model_dir,
            output_dir,
        )

        # 4. Video MCQ: 5 categories
        write_video_csvs(
            model_dir=model_dir,
            output_dir=output_dir,
        )

        # 5. Fill-in-the-blank expert evaluation template
        write_fill_blank_csv(
            model_dir=model_dir,
            output_dir=output_dir,
        )

        print(
            f"[done] {model_name} "
            f"csv_count={count_csv_files(output_dir)}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
