#!/usr/bin/env python3
"""Run long-video QA by sampling 128 frames and merging them into 32 2x2 grids."""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI
from PIL import Image

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
    parse_choice_answer,
    response_text,
    resumable_results,
    save_results,
    should_retry,
    temperature_kwargs,
)


SOURCE_FILES = [
    "video/Temporal_Sequence_Reasoning/long_Temporal_Sequence_Reasoning.json",
    "video/Long-Video_Experimental_Procedure_Reasoning/Long-Video_Experimental_Procedure_Reasoning.json",
]
VIDEO_ROOT = "video/video_data/long"
FRAME_COUNT = 128
GRID_GROUP_SIZE = 4
GRID_COUNT = FRAME_COUNT // GRID_GROUP_SIZE
CHOICES = {"A", "B", "C", "D"}

MCQ_PROMPT_TEMPLATE = """These 32 image grids come from one chemistry-lab long video in time order.
Choose one option using only the grids.
No reasoning. No explanation. Output exactly one JSON object: {{"answer":"A"}}.
Question: {question}
Options:
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}
"""

FILL_BLANK_PROMPT_TEMPLATE = """These 32 image grids come from one chemistry-lab long video in time order.
Answer all blanks in order using only the grids.
No reasoning. No explanation. Output exactly one JSON object: {{"answer":["a","b"]}}.
Question: {question}
Number of blanks: {blank_count}
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run long-video QA with 128 sampled frames merged into 32 grids."
    )
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
    add_max_output_tokens_argument(parser, default=128)
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


def format_seconds(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def probe_duration(video_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def sample_timestamps(duration: float, frame_count: int) -> list[float]:
    if duration <= 0:
        return [0.0] * frame_count
    step = duration / frame_count
    return [min(duration, step * i + step / 2) for i in range(frame_count)]


def extract_frames(video_path: Path, frame_count: int) -> tuple[Path, list[Path]]:
    duration = probe_duration(video_path)
    timestamps = sample_timestamps(duration, frame_count)
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{video_path.stem}_frames_"))
    frame_paths: list[Path] = []
    try:
        for idx, ts in enumerate(timestamps, start=1):
            out_path = temp_dir / f"frame_{idx:03d}.jpg"
            last_error: Exception | None = None
            for attempt_ts in (ts, max(0.0, ts - 0.25), max(0.0, ts - 0.5), max(0.0, ts - 1.0)):
                try:
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-y",
                            "-ss",
                            f"{attempt_ts:.3f}",
                            "-i",
                            str(video_path),
                            "-frames:v",
                            "1",
                            "-q:v",
                            "2",
                            str(out_path),
                        ],
                        check=True,
                        capture_output=True,
                    )
                    last_error = None
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
            if last_error is not None:
                raise last_error
            frame_paths.append(out_path)
        return temp_dir, frame_paths
    except Exception:  # noqa: BLE001
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def merge_four_frames_to_grid(frame_paths: list[Path], output_dir: Path) -> list[Path]:
    if len(frame_paths) != FRAME_COUNT:
        raise ValueError(f"Expected {FRAME_COUNT} frames, got {len(frame_paths)}")
    grid_paths: list[Path] = []
    for grid_idx in range(GRID_COUNT):
        group = frame_paths[grid_idx * GRID_GROUP_SIZE : (grid_idx + 1) * GRID_GROUP_SIZE]
        images = [Image.open(path).convert("RGB") for path in group]
        try:
            widths = [img.width for img in images]
            heights = [img.height for img in images]
            tile_w = min(widths)
            tile_h = min(heights)
            resized = [img.resize((tile_w, tile_h)) for img in images]
            canvas = Image.new("RGB", (tile_w * 2, tile_h * 2), color=(255, 255, 255))
            positions = [(0, 0), (tile_w, 0), (0, tile_h), (tile_w, tile_h)]
            for tile, pos in zip(resized, positions, strict=True):
                canvas.paste(tile, pos)
            grid_path = output_dir / f"grid_{grid_idx + 1:02d}.jpg"
            canvas.save(grid_path, quality=90)
            grid_paths.append(grid_path)
        finally:
            for img in images:
                img.close()
    return grid_paths


def image_to_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def build_prompt(item: dict[str, Any]) -> tuple[str, str]:
    options = item.get("options")
    if isinstance(options, list) and len(options) == 4:
        prompt = MCQ_PROMPT_TEMPLATE.format(
            question=item.get("question", ""),
            option_a=options[0],
            option_b=options[1],
            option_c=options[2],
            option_d=options[3],
        )
        return "mcq", prompt

    answer = item.get("answer")
    if isinstance(answer, list):
        prompt = FILL_BLANK_PROMPT_TEMPLATE.format(
            question=item.get("question", ""),
            blank_count=len(answer),
        )
        return "fill_blank", prompt

    raise ValueError(
        f"Unsupported long-video question format for question_id={item.get('question_id')}"
    )


def parse_model_answer(raw: str, task_kind: str) -> str | list[str] | None:
    text = raw.strip()
    if not text:
        return None
    try:
        data = load_json_from_model_text(text)
        answer = data.get("answer")
        if task_kind == "mcq":
            value = str(answer).strip().upper()
            if value in CHOICES:
                return value
        else:
            if isinstance(answer, list):
                return [str(item).strip() for item in answer]
            if isinstance(answer, str) and answer.strip():
                return [segment.strip() for segment in answer.split(",") if segment.strip()]
    except json.JSONDecodeError:
        pass

    if task_kind == "mcq":
        return parse_choice_answer(text, CHOICES)

    return [line.strip() for line in text.splitlines() if line.strip()]


def ask_model(
    client: OpenAI,
    model: str,
    prompt: str,
    grid_paths: list[Path],
    retries: int,
    task_kind: str,
    provider: str,
    max_output_tokens: int,
) -> str | list[str] | None:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for grid_path in grid_paths:
        content.append(
            {
                "type": "image_url",
                "image_url": image_url_payload(image_to_data_url(grid_path)),
            }
        )

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                **completion_token_kwargs(provider, max_output_tokens),
                **temperature_kwargs(),
                **extra_body_kwargs(),
            )
            raw = response_text(response)
            answer = parse_model_answer(raw, task_kind)
            if answer is None or answer == []:
                raise ValueError(f"Could not parse answer from model output: {raw}")
            return answer
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == retries or not should_retry(exc):
                break
            time.sleep(min(2 * attempt, 5))
    assert last_error is not None
    if isinstance(last_error, API_ERRORS):
        raise last_error
    print(f"[warn] {last_error}", file=sys.stderr, flush=True)
    return None


def process_source_file(
    client: OpenAI,
    model: str,
    source_path: Path,
    video_root: Path,
    output_path: Path,
    limit: int | None,
    retries: int,
    sleep_seconds: float,
    skip_existing: bool,
    provider: str,
    max_output_tokens: int,
    guard: ApiFailureGuard,
) -> None:
    data = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{source_path} is not a JSON array.")
    if limit is not None:
        data = data[:limit]

    print(f"[file] {source_path} items={len(data)}", flush=True)
    results: list[dict[str, Any]] = []
    completed_keys: set[str] = set()
    if output_path.exists() and skip_existing:
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if not isinstance(existing, list):
            raise ValueError(f"{output_path} is not a JSON array.")
        results = resumable_results(existing)
        completed_keys = {
            build_result_key(item.get("filename"), item.get("question"))
            for item in results
        }
    started_at = time.time()

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
        video_path = video_root / item["filename"]
        if not video_path.exists():
            raise FileNotFoundError(f"Missing video: {video_path}")
        task_kind, prompt = build_prompt(item)
        item_started_at = time.time()
        temp_dir, frame_paths = extract_frames(video_path, FRAME_COUNT)
        api_error: Exception | None = None
        try:
            grid_paths = merge_four_frames_to_grid(frame_paths, temp_dir)
            model_answer = ask_model(
                client,
                model,
                prompt,
                grid_paths,
                retries,
                task_kind,
                provider,
                max_output_tokens,
            )
        except API_ERRORS as exc:
            model_answer, api_error = None, exc
            print(f"[warn] API request failed: {exc}", file=sys.stderr, flush=True)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        row = {
            "filename": item.get("filename"),
            "question": item.get("question"),
            "answer": item.get("answer"),
            "model_answer": model_answer,
        }
        if api_error is not None:
            row["api_error"] = str(api_error)
        results.append(row)
        completed_keys.add(item_key)
        save_results(output_path, results)
        guard.record(api_error)

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
        time.sleep(sleep_seconds)

    save_results(output_path, results)
    print(
        f"[done] output={output_path} api_failures={count_api_failures(results)}",
        flush=True,
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    client = load_client(args.provider)
    model = resolve_model(args.provider, args.model)
    video_root = dataset_root / VIDEO_ROOT
    guard = ApiFailureGuard()

    print(
        f"[start] task=long_video provider={args.provider} model={model} "
        f"frames={FRAME_COUNT} grids={GRID_COUNT}",
        flush=True,
    )

    try:
        for relative in SOURCE_FILES:
            source_path = dataset_root / relative
            output_path = output_root / f"{source_path.stem}_answer.json"
            if output_path.exists() and not args.overwrite and not args.skip_existing:
                raise FileExistsError(
                    f"Output exists: {output_path}. Use --overwrite or --skip-existing."
                )
            process_source_file(
                client=client,
                model=model,
                source_path=source_path,
                video_root=video_root,
                output_path=output_path,
                limit=args.limit,
                retries=args.retries,
                sleep_seconds=args.sleep_seconds,
                skip_existing=args.skip_existing,
                provider=args.provider,
                max_output_tokens=args.max_output_tokens,
                guard=guard,
            )
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
