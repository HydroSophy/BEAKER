#!/usr/bin/env python3
"""Run all benchmark exams with one model config and isolated outputs."""

from __future__ import annotations

import argparse
import importlib.util
import os
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

from request_limits import DEFAULT_TEMPERATURE

RUNNERS = [
    "imageQAtask/run_equipment_reagent_identification.py",
    "imageQAtask/run_experimental_apparatus_setup.py",
    "imageQAtask/run_experimental_operations.py",
    "imageQAtask/run_experimental_safety_and_standardization.py",
    "imageQAtask/run_state_recognition.py",
    "affordance/run_affordance.py",
    "trajectory/run_trajectory.py",
    "short_video/run_short_video.py",
    "mid_video/run_mid_video.py",
    "long_video/run_long_video.py",
]

PLACEHOLDER_VALUES = {
    "",
    "your-api-key",
    "your-model-name",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run every benchmark runner with a model config."
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        required=True,
        help="Python config containing API_KEY, BASE_URL, MODEL, and optional OUTPUT_NAME.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Dataset root directory.",
    )
    parser.add_argument(
        "--answer-root",
        type=Path,
        default=Path(__file__).resolve().parent / "answer",
        help="Root directory for all model answer folders.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Override OUTPUT_NAME from the model config for the answer folder.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N questions in each source file.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retry count passed through to every runner.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.1,
        help="Delay between requests passed through to every runner.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow runners to overwrite existing answer files.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip questions that already exist in the model output folder.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Override per-runner maximum model output tokens.",
    )
    parser.add_argument(
        "--start-at",
        choices=RUNNERS,
        default=None,
        help="Start from a specific runner path.",
    )
    return parser


def format_seconds(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def load_config(path: Path) -> ModuleType:
    resolved = path.resolve()
    spec = importlib.util.spec_from_file_location("model_config", resolved)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load model config: {resolved}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clean_config_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def validate_config(config: ModuleType) -> tuple[str, str, str, str]:
    api_key = clean_config_value(getattr(config, "API_KEY", ""))
    base_url = clean_config_value(getattr(config, "BASE_URL", ""))
    model = clean_config_value(getattr(config, "MODEL", ""))
    output_name = clean_config_value(getattr(config, "OUTPUT_NAME", "")) or model
    if api_key in PLACEHOLDER_VALUES:
        raise RuntimeError(
            "API_KEY is missing or still uses the placeholder value. "
            "Set API_KEY in the model config."
        )
    if model in PLACEHOLDER_VALUES:
        raise RuntimeError("MODEL is missing or still uses the placeholder value.")
    return api_key, base_url, model, output_name


def api_failure_counts(output_root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in sorted(output_root.glob("*_answer.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        failed = sum(
            1 for item in data if isinstance(item, dict) and "api_error" in item
        )
        if failed:
            counts[path.name] = failed
    return counts

def safe_folder_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("._-") or "model"


def runner_subset(start_at: str | None) -> list[str]:
    if start_at is None:
        return RUNNERS
    start_index = RUNNERS.index(start_at)
    return RUNNERS[start_index:]


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    exam_root = Path(__file__).resolve().parent
    config = load_config(args.model_config)
    api_key, base_url, model, output_name = validate_config(config)
    if args.output_name:
        output_name = args.output_name
    config_max_output_tokens = getattr(config, "MAX_OUTPUT_TOKENS", None)
    max_output_tokens = args.max_output_tokens
    if max_output_tokens is None and config_max_output_tokens is not None:
        max_output_tokens = int(config_max_output_tokens)
    token_limit_param = clean_config_value(
        getattr(config, "TOKEN_LIMIT_PARAM", "")
    )
    extra_body = getattr(config, "EXTRA_BODY", None)
    temperature = getattr(config, "TEMPERATURE", DEFAULT_TEMPERATURE)
    output_root = args.answer_root.resolve() / safe_folder_name(output_name)
    output_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["OPENAI_API_KEY"] = api_key
    env["OPENAI_MODEL"] = model
    if base_url:
        env["OPENAI_BASE_URL"] = base_url
    else:
        env.pop("OPENAI_BASE_URL", None)
    if token_limit_param:
        env["OPENAI_TOKEN_LIMIT_PARAM"] = token_limit_param
    else:
        env.pop("OPENAI_TOKEN_LIMIT_PARAM", None)
    if extra_body:
        env["OPENAI_EXTRA_BODY_JSON"] = json.dumps(extra_body, ensure_ascii=False)
    else:
        env.pop("OPENAI_EXTRA_BODY_JSON", None)
    env["OPENAI_TEMPERATURE"] = "none" if temperature is None else str(float(temperature))

    selected_runners = runner_subset(args.start_at)
    print(
        "[start] "
        f"model={model} "
        f"base_url={base_url or 'OpenAI default'} "
        f"output_root={output_root} "
        f"max_output_tokens={max_output_tokens or 'runner defaults'} "
        f"temperature={'not sent' if temperature is None else temperature} "
        f"runners={len(selected_runners)}",
        flush=True,
    )

    overall_started_at = time.time()
    python_exe = sys.executable
    for index, relative_runner in enumerate(selected_runners, start=1):
        runner_path = exam_root / relative_runner
        cmd = [
            python_exe,
            str(runner_path),
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(output_root),
            "--provider",
            "openai",
            "--model",
            model,
            "--retries",
            str(args.retries),
            "--sleep-seconds",
            str(args.sleep_seconds),
        ]
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
        if args.overwrite:
            cmd.append("--overwrite")
        if args.skip_existing:
            cmd.append("--skip-existing")
        if max_output_tokens is not None:
            cmd.extend(["--max-output-tokens", str(max_output_tokens)])

        print(f"[runner {index}/{len(selected_runners)}] start {relative_runner}", flush=True)
        runner_started_at = time.time()
        completed = subprocess.run(cmd, cwd=str(dataset_root), env=env)
        runner_elapsed = time.time() - runner_started_at
        if completed.returncode != 0:
            print(
                f"[error] runner_failed {relative_runner} "
                f"elapsed={format_seconds(runner_elapsed)} "
                f"returncode={completed.returncode}",
                file=sys.stderr,
                flush=True,
            )
            return completed.returncode
        print(
            f"[runner {index}/{len(selected_runners)}] done {relative_runner} "
            f"elapsed={format_seconds(runner_elapsed)}",
            flush=True,
        )

    print(
        f"[summary] completed output_root={output_root} "
        f"elapsed={format_seconds(time.time() - overall_started_at)}",
        flush=True,
    )
    failures = api_failure_counts(output_root)
    if failures:
        details = ", ".join(f"{name}={count}" for name, count in failures.items())
        print(
            f"[warn] {sum(failures.values())} questions have no answer because the "
            f"API request failed ({details}). Rerun the same command with "
            "--skip-existing to retry them; unresolved failures count as wrong answers.",
            file=sys.stderr,
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
