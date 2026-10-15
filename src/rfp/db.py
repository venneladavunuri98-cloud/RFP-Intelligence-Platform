"""Postgres schema and connections (guide §2.2). Run once: python -m rfp.db"""
import json
from contextlib import contextmanager

import psycopg

from . import config

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS capabilities (
  id SERIAL PRIMARY KEY, section TEXT, content TEXT,
  embedding vector({config.EMBED_DIM}) -- match your embedding model's dimension
);
CREATE TABLE IF NOT EXISTS audit_llm_calls (
  id SERIAL PRIMARY KEY, run_id TEXT, node TEXT, model TEXT,
  input_tokens INT, output_tokens INT, latency_ms INT, cost_usd NUMERIC,
  created_at TIMESTAMPTZ DEFAULT now()
);
-- Additive, nullable columns (approved): retry/failure telemetry.
ALTER TABLE audit_llm_calls ADD COLUMN IF NOT EXISTS attempt INT,
  ADD COLUMN IF NOT EXISTS ok BOOLEAN, ADD COLUMN IF NOT EXISTS error TEXT;
CREATE TABLE IF NOT EXISTS reviews (
  run_id TEXT PRIMARY KEY, action TEXT, reason TEXT, reviewer TEXT,
  ai_recommendation TEXT, final_recommendation TEXT, score JSONB,
  decided_at TIMESTAMPTZ DEFAULT now()
);
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, source_file TEXT, status TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);
"""


def connect(uri: str | None = None, **kwargs) -> psycopg.Connection:
    return psycopg.connect(uri or config.DB_URI, **kwargs)


def init_db(uri: str | None = None) -> None:
    """Idempotent: safe to run on every start."""
    with connect(uri, autocommit=True) as conn:
        conn.execute(SCHEMA)


def capability_count(uri: str | None = None) -> int:
    with connect(uri) as conn:
        return conn.execute("SELECT count(*) FROM capabilities").fetchone()[0]


def insert_llm_call(rec: dict, uri: str | None = None) -> None:
    with connect(uri) as conn:
        conn.execute(
            "INSERT INTO audit_llm_calls (run_id, node, model, input_tokens, output_tokens, "
            "latency_ms, cost_usd, attempt, ok, error) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (rec["run_id"], rec["node"], rec["model"], rec["input_tokens"], rec["output_tokens"],
             rec["latency_ms"], rec["cost_usd"], rec.get("attempt"), rec.get("ok"), rec.get("error")))


def upsert_run(run_id: str, status: str, source_file: str | None = None,
               uri: str | None = None) -> None:
    with connect(uri) as conn:
        conn.execute(
            "INSERT INTO runs (run_id, source_file, status) VALUES (%s, %s, %s) "
            "ON CONFLICT (run_id) DO UPDATE SET status = EXCLUDED.status, "
            "source_file = COALESCE(EXCLUDED.source_file, runs.source_file)",
            (run_id, source_file, status))


def write_review(run_id: str, review: dict, score: dict | None, uri: str | None = None) -> None:
    """Idempotent: finalize may re-run after a crash, so upsert by run_id."""
    with connect(uri) as conn:
        conn.execute(
            "INSERT INTO reviews (run_id, action, reason, reviewer, ai_recommendation, "
            "final_recommendation, score, decided_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s) "
            "ON CONFLICT (run_id) DO UPDATE SET action = EXCLUDED.action, reason = EXCLUDED.reason, "
            "reviewer = EXCLUDED.reviewer, ai_recommendation = EXCLUDED.ai_recommendation, "
            "final_recommendation = EXCLUDED.final_recommendation, score = EXCLUDED.score, "
            "decided_at = EXCLUDED.decided_at",
            (run_id, review["action"], review.get("reason") or "", review["reviewer"],
             review.get("ai_recommendation"), review["final_recommendation"],
             json.dumps(score) if score is not None else None, review["decided_at"]))


def runs_with_status(status: str, uri: str | None = None) -> list[dict]:
    with connect(uri) as conn:
        rows = conn.execute("SELECT run_id, source_file, status, created_at FROM runs "
                            "WHERE status = %s ORDER BY created_at DESC", (status,)).fetchall()
    return [{"run_id": r, "source_file": f, "status": s, "created_at": c.isoformat()}
            for r, f, s, c in rows]


def run_telemetry(run_id: str, uri: str | None = None) -> list[dict]:
    cols = ("node", "attempt", "ok", "input_tokens", "output_tokens", "latency_ms", "cost_usd")
    with connect(uri) as conn:
        rows = conn.execute(f"SELECT {', '.join(cols)} FROM audit_llm_calls WHERE run_id = %s "
                            "ORDER BY id", (run_id,)).fetchall()
    return [dict(zip(cols, (*r[:6], float(r[6] or 0)))) for r in rows]


def telemetry_summary(uri: str | None = None) -> list[dict]:
    """Per-run LLM totals across all runs (newest first)."""
    with connect(uri) as conn:
        rows = conn.execute(
            "SELECT run_id, count(*), sum(input_tokens), sum(output_tokens), sum(latency_ms), "
            "sum(cost_usd), bool_and(coalesce(ok, true)), min(created_at) FROM audit_llm_calls "
            "GROUP BY run_id ORDER BY min(created_at) DESC").fetchall()
    return [{"run_id": r, "calls": n, "input_tokens": int(i or 0), "output_tokens": int(o or 0),
             "latency_s": round((l or 0) / 1000, 1), "cost_usd": float(c or 0), "all_ok": ok,
             "started_at": t.isoformat()} for r, n, i, o, l, c, ok, t in rows]


@contextmanager
def checkpointer(uri: str | None = None, max_size: int = 10):
    """PostgresSaver on a connection pool (guide §2.4); creates its tables once."""
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool
    with ConnectionPool(uri or config.DB_URI, min_size=1, max_size=max_size, open=True,
                        kwargs={"autocommit": True, "prepare_threshold": 0,
                                "row_factory": dict_row}) as pool:
        cp = PostgresSaver(pool)
        cp.setup()
        yield cp


if __name__ == "__main__":
    init_db()
    print("schema ready")
