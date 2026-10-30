#!/usr/bin/env python3
"""Run image QA for state recognition."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from exam_common import run_from_args


if __name__ == "__main__":
    raise SystemExit(
        run_from_args(
            "state_recognition",
            "Run state recognition image exam.",
        )
    )
