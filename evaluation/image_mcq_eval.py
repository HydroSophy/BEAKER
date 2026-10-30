"""Answer files and scoring for the five image multiple-choice tasks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CATEGORIES = {
    "Equipment_and_Reagent_Identification": {
        "simple": "simple_Equipment_and_Reagent_Identification_answer.json",
        "complex": "complex_Equipment_and_Reagent_Identification_answer.json",
        "output": "equipment_and_reagent_identification_scores.csv",
    },
    "Experimental_Apparatus_Setup": {
        "simple": "simple_Experimental_Apparatus_Setup_answer.json",
        "complex": "complex_Experimental_Apparatus_Setup_answer.json",
        "output": "experimental_apparatus_setup_scores.csv",
    },
    "Experimental_Operations": {
        "simple": "simple_Experimental_Operations_answer.json",
        "complex": "complex_Experimental_Operations_answer.json",
        "output": "experimental_operations_scores.csv",
    },
    "Experimental_Safety_and_Standardization": {
        "simple": "simple_Experimental_Safety_and_Standardization_answer.json",
        "complex": "complex_Experimental_Safety_and_Standardization_answer.json",
        "output": "experimental_safety_and_standardization_scores.csv",
    },
    "State_Recognition": {
        "simple": "simple_State_Recognition_answer.json",
        "complex": "complex_State_Recognition_answer.json",
        "output": "state_recognition_scores.csv",
    },
}


def load_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON array.")
    return data


def normalize_answer(value: Any) -> str:
    return str(value or "").strip().upper()


def build_rows(data: list[dict[str, Any]], difficulty: str) -> tuple[int, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    correct = 0
    for index, item in enumerate(data, start=1):
        answer = normalize_answer(item.get("answer"))
        model_answer = normalize_answer(item.get("model_answer"))
        score = 1 if answer and answer == model_answer else 0
        correct += score
        rows.append(
            {
                "difficulty": difficulty,
                "index_in_split": index,
                "filename": item.get("filename"),
                "question": item.get("question"),
                "answer": answer,
                "model_answer": model_answer,
                "score": score,
            }
        )
    return correct, rows
