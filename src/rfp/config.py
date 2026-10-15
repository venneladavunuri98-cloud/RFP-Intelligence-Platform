"""Runtime configuration from environment variables (never hardcode secrets).

GOOGLE_API_KEY is read directly by langchain-google-genai, not here.
"""
import json
import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]


def load_dotenv(path: pathlib.Path) -> None:
    """Minimal KEY=VALUE .env loader; real environment variables take precedence."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


load_dotenv(ROOT / ".env")

# init_chat_model id, e.g. "google_genai:<model-id>"
MODEL = os.environ.get("LLM_MODEL", "")

# USD per 1M tokens keyed by model id, e.g. '{"google_genai:x": {"in": 0.3, "out": 2.5}}'.
# Free tier: leave empty -> cost logged as 0.
PRICES: dict = json.loads(os.environ.get("LLM_PRICES_JSON", "{}"))

EXTRACT_MAX_ATTEMPTS = 3  # 1 try + 2 retries (guide §1.5)
RETRY_BACKOFF_S = float(os.environ.get("RETRY_BACKOFF_S", "5"))  # free-tier 429s

# "postgres": checkpoints, telemetry and audit rows in Postgres (Stage 2 default).
# "local": MemorySaver + JSONL telemetry, no DB writes (unit tests, offline use).
STORAGE = os.environ.get("RFP_STORAGE", "postgres")

# Postgres + pgvector (Stage 2). Local-dev default matches docker-compose.yml.
DB_URI = os.environ.get("DB_URI", "postgresql://rfp:rfp@127.0.0.1:5432/rfp")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "models/gemini-embedding-001")
EMBED_DIM = 1536  # must equal capabilities.embedding vector(N) in db.py

# Stage 3: API, uploads, notifications
REPORT_DIR = pathlib.Path(os.environ.get("REPORT_DIR", ROOT / "reports"))
UPLOAD_DIR = pathlib.Path(os.environ.get("UPLOAD_DIR", ROOT / "data" / "uploads"))
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "25"))
N8N_WEBHOOK_URL = os.environ.get("N8N_WEBHOOK_URL", "")  # empty = notifications disabled

TELEMETRY_LOG = pathlib.Path(os.environ.get("TELEMETRY_LOG", ROOT / "audit_llm_calls.jsonl"))
RUBRIC_PATH = ROOT / "rubric.yaml"
