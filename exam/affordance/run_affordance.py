#!/usr/bin/env python3
"""Run Affordance Grounding (AFF) point prediction."""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    load_json_from_model_text,
    response_text,
    resumable_results,
    save_results,
    should_retry,
    temperature_kwargs,
)


PROMPT_TEMPLATE = """Find the best point for this chemistry-lab image question.
Use image coordinates with top-left as (0,0), X rightward, Y downward.
Coordinates must be integers within width={width}, height={height}.
No reasoning. No explanation. Output exactly one JSON object: {{"X":123,"Y":456}}.
Question: {question}
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run affordance point grounding.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent,
        help="Dataset root directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "answer",
        help="Directory used to store answer outputs.",
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
        help="Only run the first N questions.",
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
    add_max_output_tokens_argument(parser, default=32)
    return parser


def build_result_key(filename: Any, question: Any) -> str:
    return f"{filename}\n{question}"


def resolve_model(provider: str, override: str | None) -> str:
    if override:
        return override
    if provider == "openai":
        return DEFAULT_OPENAI_MODEL
    return DEFAULT_LOCAL_MODEL


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


def clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def center_of_box(box: dict[str, Any]) -> dict[str, int]:
    x = round((int(box["x1"]) + int(box["x2"])) / 2)
    y = round((int(box["y1"]) + int(box["y2"])) / 2)
    return {"x": x, "y": y}


def parse_point_response(text: str, width: int, height: int) -> dict[str, int] | None:
    raw = text.strip()
    if not raw:
        return None
    try:
        data = load_json_from_model_text(raw)
        if "X" in data and "Y" in data:
            x = int(round(float(data["X"])))
            y = int(round(float(data["Y"])))
        else:
            x = int(round(float(data["x"])))
            y = int(round(float(data["y"])))
        return {
            "x": clamp(x, 0, width),
            "y": clamp(y, 0, height),
        }
    except Exception:  # noqa: BLE001
        pass

    x_match = re.search(r"['\"]?[Xx]['\"]?\s*:\s*(-?\d+(?:\.\d+)?)", raw)
    y_match = re.search(r"['\"]?[Yy]['\"]?\s*:\s*(-?\d+(?:\.\d+)?)", raw)
    if x_match and y_match:
        x = int(round(float(x_match.group(1))))
        y = int(round(float(y_match.group(1))))
        return {
            "x": clamp(x, 0, width),
            "y": clamp(y, 0, height),
        }

    match = re.search(r"(-?\d+(?:\.\d+)?)\s*[,，]\s*(-?\d+(?:\.\d+)?)", raw)
    if match:
        x = int(round(float(match.group(1))))
        y = int(round(float(match.group(2))))
        return {
            "x": clamp(x, 0, width),
            "y": clamp(y, 0, height),
        }
    return None


def ask_model(
    client: OpenAI,
    model: str,
    prompt: str,
    image_path: Path,
    width: int,
    height: int,
    retries: int,
    provider: str,
    max_output_tokens: int,
) -> tuple[dict[str, int] | None, str]:
    data_url = image_to_data_url(image_path)
    last_error: Exception | None = None
    last_content = ""
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
            last_content = content
            point = parse_point_response(content, width, height)
            if point is None:
                raise ValueError(f"Could not parse point from model output: {content}")
            return point, content
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == retries or not should_retry(exc):
                break
            time.sleep(min(2 * attempt, 5))
    assert last_error is not None
    if isinstance(last_error, API_ERRORS):
        raise last_error
    print(f"[warn] {last_error}", file=sys.stderr, flush=True)
    return None, last_content


def format_seconds(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    model = resolve_model(args.provider, args.model)
    client = load_client(args.provider)

    source_path = dataset_root / "image" / "Affordance_Grounding" / "Affordance_Grounding.json"
    image_root = dataset_root / "image" / "image_data" / "affordance"
    output_path = output_root / "affordance_answer.json"
    output_root.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.overwrite and not args.skip_existing:
        print(
            f"[error] Output exists: {output_path}. Use --overwrite or --skip-existing.",
            file=sys.stderr,
        )
        return 1

    data = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        print(f"[error] {source_path.name} is not a JSON array.", file=sys.stderr)
        return 1
    if args.limit is not None:
        data = data[: args.limit]

    print(
        f"[start] task=affordance provider={args.provider} model={model} items={len(data)}",
        flush=True,
    )

    results: list[dict[str, Any]] = []
    completed_keys: set[str] = set()
    if output_path.exists() and args.skip_existing:
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if not isinstance(existing, list):
            print("[error] Existing affordance answer file is not a JSON array.", file=sys.stderr)
            return 1
        results = resumable_results(existing)
        completed_keys = {
            build_result_key(item.get("filename"), item.get("question"))
            for item in results
        }
    started_at = time.time()
    guard = ApiFailureGuard()

    for index, item in enumerate(data, start=1):
        item_key = build_result_key(item.get("filename"), item.get("question"))
        if item_key in completed_keys:
            elapsed = time.time() - started_at
            avg = elapsed / index if index else 0
            remaining = len(data) - index
            eta = avg * remaining
            print(
                "[progress] "
                f"item={index}/{len(data)} "
                f"status=skipped "
                f"elapsed={format_seconds(elapsed)} "
                f"eta={format_seconds(eta)}",
                flush=True,
            )
            continue
        image_path = image_root / item["filename"]
        if not image_path.exists():
            print(f"[error] Missing image: {image_path}", file=sys.stderr)
            return 1

        width = int(item["image_size"]["width"])
        height = int(item["image_size"]["height"])
        prompt = PROMPT_TEMPLATE.format(
            width=width,
            height=height,
            question=item["question"],
        )
        item_started_at = time.time()
        api_error: Exception | None = None
        try:
            model_answer, raw_output = ask_model(
                client,
                model,
                prompt,
                image_path,
                width,
                height,
                args.retries,
                args.provider,
                args.max_output_tokens,
            )
        except API_ERRORS as exc:
            model_answer, raw_output, api_error = None, "", exc
            print(f"[warn] API request failed: {exc}", file=sys.stderr, flush=True)
        # Keep one representative point in answer; evaluation uses every box in answers.
        gold_answer = center_of_box(item["answers"][0])

        row = {
            "question_id": item.get("question_id"),
            "filename": item["filename"],
            "question": item["question"],
            "image_size": item["image_size"],
            "answers": item["answers"],
            "answer": gold_answer,
            "model_answer": model_answer,
            "raw_output": raw_output,
        }
        if api_error is not None:
            row["api_error"] = str(api_error)
        results.append(row)
        completed_keys.add(item_key)
        save_results(output_path, results)
        try:
            guard.record(api_error)
        except RuntimeError as exc:
            print(f"[error] {exc}", file=sys.stderr)
            return 1

        elapsed = time.time() - started_at
        avg = elapsed / index if index else 0
        remaining = len(data) - index
        eta = avg * remaining
        print(
            "[progress] "
            f"item={index}/{len(data)} "
            f"item_time={format_seconds(time.time() - item_started_at)} "
            f"elapsed={format_seconds(elapsed)} "
            f"eta={format_seconds(eta)}",
            flush=True,
        )
        time.sleep(args.sleep_seconds)

    save_results(output_path, results)
    print(
        f"[done] output={output_path} api_failures={count_api_failures(results)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
