"""Shared request helpers for exam runners."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from openai import APIError, APIStatusError


DEFAULT_TEMPERATURE = 0.7

# Sending the same request again cannot change the outcome.
NON_RETRYABLE_STATUS_CODES = {400, 401, 403, 404, 422}
# The API key, base URL, or model name is wrong; every later request would fail too.
CONFIGURATION_STATUS_CODES = {401, 403, 404}
MAX_CONSECUTIVE_API_FAILURES = 5


class EmptyResponseError(RuntimeError):
    """The endpoint returned no choices, e.g. a proxy that reports errors with HTTP 200."""


API_ERRORS = (APIError, EmptyResponseError)


def response_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        detail = getattr(response, "model_extra", None) or response
        raise EmptyResponseError(f"Response contains no choices: {detail}")
    return choices[0].message.content or ""


def should_retry(exc: Exception) -> bool:
    return not (
        isinstance(exc, APIStatusError)
        and exc.status_code in NON_RETRYABLE_STATUS_CODES
    )


class ApiFailureGuard:
    """Stop the run after `limit` consecutive failures, or at once on a configuration error."""

    def __init__(self, limit: int = MAX_CONSECUTIVE_API_FAILURES) -> None:
        self.limit = limit
        self.consecutive = 0

    def record(self, error: Exception | None) -> None:
        if error is None:
            self.consecutive = 0
            return
        self.consecutive += 1
        if (
            isinstance(error, APIStatusError)
            and error.status_code in CONFIGURATION_STATUS_CODES
        ):
            raise RuntimeError(
                f"API request rejected; check API_KEY, BASE_URL, and MODEL: {error}"
            ) from error
        if self.consecutive >= self.limit:
            raise RuntimeError(
                f"{self.consecutive} consecutive API requests failed; stopping. "
                f"Last error: {error}"
            ) from error


def resumable_results(existing: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep answered items; items whose request failed are asked again."""

    return [item for item in existing if "api_error" not in item]


def count_api_failures(results: list[dict[str, Any]]) -> int:
    return sum(1 for item in results if "api_error" in item)


def save_results(path: Path, results: list[dict[str, Any]]) -> None:
    path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

ANSWER_CHOICE_PATTERN = re.compile(
    r"(?:final\s+answer|answer|correct\s+(?:answer|option|choice|order|sequence)|"
    r"correct|答案|选择)"
    r"\s*(?:is|为|是|:|：)?\s*(?:option|choice|选项)?\s*[\(\[]?\s*([A-D])\b",
    flags=re.IGNORECASE,
)
OPTION_CHOICE_PATTERN = re.compile(
    r"(?:选项|option|choice|choose)"
    r"\s*(?:is|为|是|:|：)?\s*(?:option|choice|选项)?\s*[\(\[]?\s*([A-D])\b",
    flags=re.IGNORECASE,
)


def add_max_output_tokens_argument(
    parser: argparse.ArgumentParser,
    default: int,
) -> None:
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=default,
        help="Maximum model output tokens for each request.",
    )


def image_url_payload(data_url: str) -> dict[str, str]:
    """Build an image payload without provider-specific quality controls."""

    return {"url": data_url}


def completion_token_kwargs(provider: str, max_output_tokens: int | None) -> dict[str, Any]:
    if max_output_tokens is None or max_output_tokens <= 0:
        return {}
    if os.environ.get("OPENAI_TOKEN_LIMIT_PARAM", "").strip() == "max_tokens":
        return {"max_tokens": max_output_tokens}
    if provider == "openai":
        return {"max_completion_tokens": max_output_tokens}
    return {"max_tokens": max_output_tokens}


def temperature_kwargs() -> dict[str, Any]:
    """Sampling temperature; OPENAI_TEMPERATURE=none omits it for endpoints that reject it."""

    raw = os.environ.get("OPENAI_TEMPERATURE", "").strip()
    if raw.lower() == "none":
        return {}
    return {"temperature": float(raw) if raw else DEFAULT_TEMPERATURE}


def extra_body_kwargs() -> dict[str, Any]:
    raw = os.environ.get("OPENAI_EXTRA_BODY_JSON", "").strip()
    if not raw:
        return {}
    return {"extra_body": json.loads(raw)}


def strip_markdown_code_fence(text: str) -> str:
    raw = text.strip()
    match = re.fullmatch(r"```(?:json|JSON)?\s*(.*?)\s*```", raw, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw


def load_json_from_model_text(text: str) -> Any:
    raw = strip_markdown_code_fence(text)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    object_start = raw.find("{")
    object_end = raw.rfind("}")
    if object_start != -1 and object_end > object_start:
        return json.loads(raw[object_start : object_end + 1])

    array_start = raw.find("[")
    array_end = raw.rfind("]")
    if array_start != -1 and array_end > array_start:
        return json.loads(raw[array_start : array_end + 1])

    raise json.JSONDecodeError("No JSON object or array found", raw, 0)


def parse_choice_answer(text: str, choices: set[str] | None = None) -> str | None:
    allowed = choices or {"A", "B", "C", "D"}
    raw = strip_markdown_code_fence(text).strip()
    if not raw:
        return None

    try:
        data = load_json_from_model_text(raw)
        if isinstance(data, dict):
            answer = str(data.get("answer", "")).strip().upper()
            if answer in allowed:
                return answer
            match = ANSWER_CHOICE_PATTERN.search(answer) or OPTION_CHOICE_PATTERN.search(answer)
            if match and match.group(1).upper() in allowed:
                return match.group(1).upper()
    except json.JSONDecodeError:
        pass

    simple = raw.strip().upper()
    if re.fullmatch(r"[\(\[]?\s*([A-D])\s*[\)\].。．:：]?", simple):
        value = re.search(r"([A-D])", simple)
        if value and value.group(1) in allowed:
            return value.group(1)

    match = ANSWER_CHOICE_PATTERN.search(raw) or OPTION_CHOICE_PATTERN.search(raw)
    if match:
        value = match.group(1).upper()
        if value in allowed:
            return value
    return None
