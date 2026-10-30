"""Hosted OpenAI-compatible endpoint; credentials come from the environment."""
import os

API_KEY = os.environ.get("OPENAI_API_KEY", "your-api-key")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "")
MODEL = os.environ.get("OPENAI_MODEL", "your-model-name")
OUTPUT_NAME = os.environ.get("BEAKER_OUTPUT_NAME", MODEL)
MAX_OUTPUT_TOKENS = 4096
# Set to None for endpoints that reject a temperature parameter.
TEMPERATURE = 0.7
EXTRA_BODY = {}
