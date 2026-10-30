#!/usr/bin/env python3
"""Shared utilities for running image-based QA exams."""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

from model_settings import (
    DEFAULT_LOCAL_MODEL,
    DEFAULT_OPENAI_MODEL,
    LOCAL_API_KEY,
    LOCAL_BASE_URL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    make_openai_client,
)
from request_limits import (
    API_ERRORS,
    ApiFailureGuard,
    add_max_output_tokens_argument,
    completion_token_kwargs,
    count_api_failures,
    extra_body_kwargs,
    image_url_payload,
    parse_choice_answer,
    response_text,
    resumable_results,
    save_results,
    should_retry,
    temperature_kwargs,
)


TASKS: dict[str, dict[str, Any]] = {
    "equipment_reagent_identification": {
        "label": "Equipment and Reagent Identification",
        "files": [
            "image/Equipment_and_Reagent_Identification/complex_Equipment_and_Reagent_Identification.json",
            "image/Equipment_and_Reagent_Identification/simple_Equipment_and_Reagent_Identification.json",
        ],
    },
    "experimental_apparatus_setup": {
        "label": "Experimental Apparatus Setup",
        "files": [
            "image/Experimental_Apparatus_Setup/complex_Experimental_Apparatus_Setup.json",
            "image/Experimental_Apparatus_Setup/simple_Experimental_Apparatus_Setup.json",
        ],
    },
    "experimental_operations": {
        "label": "Experimental Operations",
        "files": [
            "image/Experimental_Operations/complex_Experimental_Operations.json",
            "image/Experimental_Operations/simple_Experimental_Operations.json",
        ],
    },
    "experimental_safety_and_standardization": {
        "label": "Experimental Safety and Standardization",
        "files": [
            "image/Experimental_Safety_and_Standardization/complex_Experimental_Safety_and_Standardization.json",
            "image/Experimental_Safety_and_Standardization/simple_Experimental_Safety_and_Standardization.json",
        ],
    },
    "state_recognition": {
        "label": "State Recognition",
        "files": [
            "image/State_Recognition/complex_State_Recognition.json",
            "image/State_Recognition/simple_State_Recognition.json",
        ],
    },
}

CHOICES = ["A", "B", "C", "D"]
PROMPT_TEMPLATE = """Answer using only the image and options.
No reasoning. No explanation. Output exactly one JSON object: {{"answer":"A"}}.
Question: {question}
Options:
{options_block}
"""


@dataclass
class RunConfig:
    dataset_root: Path
    output_root: Path
    task_key: str
    provider: str
    model: str
    limit: int | None
    retries: int
    sleep_seconds: float
    overwrite: bool
    skip_existing: bool
    max_output_tokens: int


def format_seconds(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Dataset root directory. Defaults to the repo dataset root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "answer",
        help="Directory used to store exam outputs.",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "local"],
        default="openai",
        help="Model provider.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name override.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N questions from each source file.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=1,
        help="Retry count for failed API calls.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.1,
        help="Delay between requests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output file.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip questions that already exist in the output file.",
    )
    add_max_output_tokens_argument(parser, default=16)
    return parser


def load_client(provider: str) -> OpenAI:
    if provider == "openai":
        return make_openai_client(OPENAI_API_KEY, OPENAI_BASE_URL)
    if provider == "local":
        if not LOCAL_BASE_URL.strip():
            raise RuntimeError("Missing LOCAL_BASE_URL in exam/model_settings.py.")
        return OpenAI(
            api_key=LOCAL_API_KEY.strip() or "EMPTY",
            base_url=LOCAL_BASE_URL.strip(),
        )
    raise ValueError(f"Unsupported provider: {provider}")


def resolve_model(provider: str, override: str | None) -> str:
    if override:
        return override
    if provider == "openai":
        return DEFAULT_OPENAI_MODEL
    return DEFAULT_LOCAL_MODEL


def resolve_image_path(dataset_root: Path, filename: str) -> Path:
    prefix = filename[:2].lower()
    if prefix == "c_":
        subdir = "complex"
    elif prefix == "s_":
        subdir = "simple"
    else:
        raise ValueError(f"Cannot infer image split from filename: {filename}")
    path = dataset_root / "image" / "image_data" / subdir / filename
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return path


def image_to_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(suffix, "application/octet-stream")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def build_prompt(item: dict[str, Any]) -> str:
    options = item.get("options", [])
    if len(options) not in {2, 3, 4}:
        raise ValueError(
            f"Question {item.get('question_id')} does not have 2-4 options."
        )
    options_block = "\n".join(
        f"{CHOICES[idx]}. {option}" for idx, option in enumerate(options)
    )
    return PROMPT_TEMPLATE.format(
        question_type=item.get("question_type", ""),
        question=item.get("question", ""),
        options_block=options_block,
    )


def parse_model_answer(text: str) -> tuple[str | None, str]:
    raw = text.strip()
    return parse_choice_answer(raw, set(CHOICES)), raw


def ask_model(
    client: OpenAI,
    provider: str,
    model: str,
    prompt: str,
    image_path: Path,
    retries: int,
    max_output_tokens: int,
) -> tuple[str | None, str]:
    data_url = image_to_data_url(image_path)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": image_url_payload(data_url),
                            },
                        ],
                    }
                ],
                **completion_token_kwargs(provider, max_output_tokens),
                **temperature_kwargs(),
                **extra_body_kwargs(),
            )
            content = response_text(response)
            answer, raw = parse_model_answer(content)
            if answer is None:
                raise ValueError(f"Could not parse choice from model output: {content}")
            return answer, raw
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == retries or not should_retry(exc):
                break
            time.sleep(min(2 * attempt, 5))
    assert last_error is not None
    if isinstance(last_error, API_ERRORS):
        raise last_error
    print(f"[warn] {last_error}", file=sys.stderr, flush=True)
    return None, ""


def load_questions(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON array.")
    return data


def build_output_path(
    output_root: Path,
    source_path: Path,
) -> Path:
    return output_root / f"{source_path.stem}_answer.json"


def build_result_key(filename: Any, question: Any) -> str:
    return f"{filename}\n{question}"


def run_task(config: RunConfig) -> int:
    task = TASKS[config.task_key]
    client = load_client(config.provider)
    source_paths = [config.dataset_root / relative for relative in task["files"]]
    total_items = 0
    for path in source_paths:
        total_items += len(load_questions(path))

    print(
        f"[start] task={config.task_key} provider={config.provider} "
        f"model={config.model} files={len(source_paths)} total_items={total_items}",
        flush=True,
    )

    global_completed = 0
    global_started_at = time.time()
    guard = ApiFailureGuard()

    for file_index, source_path in enumerate(source_paths, start=1):
        questions = load_questions(source_path)
        if config.limit is not None:
            questions = questions[: config.limit]
        output_path = build_output_path(
            config.output_root,
            source_path,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and not config.overwrite and not config.skip_existing:
            raise FileExistsError(
                f"Output exists: {output_path}. Use --overwrite or --skip-existing."
            )

        print(
            f"[file {file_index}/{len(source_paths)}] {source_path} items={len(questions)}",
            flush=True,
        )

        results: list[dict[str, Any]] = []
        completed_keys: set[str] = set()
        if output_path.exists() and config.skip_existing:
            results = resumable_results(load_questions(output_path))
            completed_keys = {
                build_result_key(item.get("filename"), item.get("question"))
                for item in results
            }
        correct_count = sum(
            1
            for item in results
            if item.get("model_answer") in CHOICES
            and item.get("model_answer") == item.get("answer")
        )

        for item_index, item in enumerate(questions, start=1):
            item_key = build_result_key(item.get("filename"), item.get("question"))
            if item_key in completed_keys:
                global_completed += 1
                elapsed = time.time() - global_started_at
                avg = elapsed / global_completed if global_completed else 0
                remaining = total_items - global_completed
                eta = avg * remaining
                print(
                    "[progress] "
                    f"task={config.task_key} "
                    f"file_item={item_index}/{len(questions)} "
                    f"global={global_completed}/{total_items} "
                    f"status=skipped "
                    f"elapsed={format_seconds(elapsed)} "
                    f"eta={format_seconds(eta)}",
                    flush=True,
                )
                continue
            item_started_at = time.time()
            image_path = resolve_image_path(config.dataset_root, item["filename"])
            prompt = build_prompt(item)
            api_error: Exception | None = None
            try:
                predicted_answer, raw_output = ask_model(
                    client,
                    config.provider,
                    config.model,
                    prompt,
                    image_path,
                    config.retries,
                    config.max_output_tokens,
                )
            except API_ERRORS as exc:
                predicted_answer, api_error = None, exc
                print(f"[warn] API request failed: {exc}", file=sys.stderr, flush=True)
            gold_answer = str(item.get("answer", "")).strip().upper()
            is_correct = predicted_answer == gold_answer and predicted_answer in CHOICES
            if is_correct:
                correct_count += 1
            global_completed += 1

            row = {
                "filename": item.get("filename"),
                "question": item.get("question"),
                "answer": gold_answer,
                "model_answer": predicted_answer,
            }
            if api_error is not None:
                row["api_error"] = str(api_error)
            results.append(row)
            save_results(output_path, results)
            guard.record(api_error)

            elapsed = time.time() - global_started_at
            avg = elapsed / global_completed if global_completed else 0
            remaining = total_items - global_completed
            eta = avg * remaining
            print(
                "[progress] "
                f"task={config.task_key} "
                f"file_item={item_index}/{len(questions)} "
                f"global={global_completed}/{total_items} "
                f"acc={correct_count}/{item_index} "
                f"item_time={format_seconds(time.time() - item_started_at)} "
                f"elapsed={format_seconds(elapsed)} "
                f"eta={format_seconds(eta)}",
                flush=True,
            )
            time.sleep(config.sleep_seconds)

        save_results(output_path, results)
        print(
            f"[done] output={output_path} accuracy={correct_count}/{len(questions)} "
            f"api_failures={count_api_failures(results)}",
            flush=True,
        )

    return 0


def run_from_args(task_key: str, description: str) -> int:
    parser = build_parser(description)
    args = parser.parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    model = resolve_model(args.provider, args.model)
    config = RunConfig(
        dataset_root=dataset_root,
        output_root=output_root,
        task_key=task_key,
        provider=args.provider,
        model=model,
        limit=args.limit,
        retries=args.retries,
        sleep_seconds=args.sleep_seconds,
        overwrite=args.overwrite,
        skip_existing=args.skip_existing,
        max_output_tokens=args.max_output_tokens,
    )
    try:
        return run_task(config)
    except Exception as exc:  # noqa: BLE001
        print(f"[error] {exc}", file=sys.stderr)
        return 1
