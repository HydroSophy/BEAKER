"""Model configuration for exam runners."""

from __future__ import annotations

import os
from typing import Any

from openai import OpenAI


def _optional_env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name, "")
    if value.strip():
        return value.strip()
    return default


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = _optional_env("OPENAI_BASE_URL")

# For local OpenAI-compatible servers such as vLLM / LM Studio / Ollama gateway.
LOCAL_API_KEY = os.environ.get("LOCAL_API_KEY", "EMPTY")
LOCAL_BASE_URL = os.environ.get("LOCAL_BASE_URL", "http://127.0.0.1:8000/v1")

# Default models. Adjust these to your actual deployment names.
DEFAULT_OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.5")
DEFAULT_LOCAL_MODEL = os.environ.get("LOCAL_MODEL", "Qwen2.5-VL-7B-Instruct")


def make_openai_client(api_key: str, base_url: str | None) -> OpenAI:
    if not api_key.strip():
        raise RuntimeError("Missing OPENAI_API_KEY.")
    kwargs: dict[str, Any] = {"api_key": api_key.strip()}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)
