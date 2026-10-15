import pymupdf


def make_pdf(path, page_texts):
    """Build a test PDF; an empty string yields a page with no text layer."""
    doc = pymupdf.open()
    for text in page_texts:
        page = doc.new_page()
        if text:
            page.insert_textbox(pymupdf.Rect(50, 50, 550, 800), text, fontsize=10)
    doc.save(path)
    doc.close()
    return str(path)


# ---- Postgres fixtures (Stage 2). Tests use a separate rfp_test database. ----
import os

import pytest

TEST_DB_URI = os.environ.get("TEST_DB_URI", "postgresql://rfp:rfp@127.0.0.1:5432/rfp_test")


@pytest.fixture(scope="session")
def pg_uri():
    """Creates (once) and returns an isolated test database; skips if Postgres is down."""
    psycopg = pytest.importorskip("psycopg")
    admin = TEST_DB_URI.rsplit("/", 1)[0] + "/postgres"
    try:
        with psycopg.connect(admin, autocommit=True, connect_timeout=3) as conn:
            if not conn.execute("SELECT 1 FROM pg_database WHERE datname = 'rfp_test'").fetchone():
                conn.execute("CREATE DATABASE rfp_test")
    except psycopg.OperationalError as e:
        pytest.skip(f"Postgres not available ({e}); run: docker compose up -d db")
    from rfp.db import init_db
    init_db(TEST_DB_URI)
    return TEST_DB_URI


@pytest.fixture(autouse=True)
def local_storage(monkeypatch, tmp_path):
    """Unit tests default to MemorySaver + JSONL; DB tests opt in via `pg_storage`."""
    from rfp import config
    monkeypatch.setattr(config, "STORAGE", "local")
    monkeypatch.setattr(config, "REPORT_DIR", tmp_path / "reports")


@pytest.fixture
def pg_storage(pg_uri, monkeypatch, tmp_path):
    from rfp import config
    monkeypatch.setattr(config, "STORAGE", "postgres")
    monkeypatch.setattr(config, "DB_URI", pg_uri)
    monkeypatch.setattr(config, "TELEMETRY_LOG", tmp_path / "fallback.jsonl")
    return pg_uri


@pytest.fixture(autouse=True)
def no_live_llm(monkeypatch, request):
    """Unit tests never call real LLM/embedding APIs (cost, quota, flakiness)."""
    if request.node.get_closest_marker("live"):
        return
    from rfp import llm

    def blocked(*a, **k):
        raise RuntimeError("live LLM call blocked in unit tests")
    monkeypatch.setattr(llm, "chat_model", blocked)
    monkeypatch.setattr(llm, "embedder", blocked)
