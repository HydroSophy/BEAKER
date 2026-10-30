"""Configure a running local OpenAI-compatible vision-language server."""
import os

API_KEY = os.environ.get("LOCAL_API_KEY", "EMPTY")
BASE_URL = os.environ.get("LOCAL_BASE_URL", "http://127.0.0.1:8000/v1")
MODEL = os.environ.get("LOCAL_MODEL", "chemvlm-8b")
OUTPUT_NAME = "local_model"
MAX_OUTPUT_TOKENS = 4096
TOKEN_LIMIT_PARAM = "max_tokens"
# Set to None for endpoints that reject a temperature parameter.
TEMPERATURE = 0.7
EXTRA_BODY = {}
