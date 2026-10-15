import psycopg

from rfp import config
from rfp.db import init_db

TABLES = {"capabilities", "audit_llm_calls", "reviews", "runs"}


def test_schema_creates_all_tables(pg_uri):
    with psycopg.connect(pg_uri) as conn:
        rows = conn.execute("SELECT table_name FROM information_schema.tables "
                            "WHERE table_schema = 'public'").fetchall()
    assert TABLES <= {r[0] for r in rows}


def test_init_db_is_idempotent(pg_uri):
    init_db(pg_uri)
    init_db(pg_uri)


def test_embedding_column_matches_configured_dimension(pg_uri):
    with psycopg.connect(pg_uri) as conn:
        typ = conn.execute("SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
                           "WHERE attrelid = 'capabilities'::regclass AND attname = 'embedding'"
                           ).fetchone()[0]
    assert typ == f"vector({config.EMBED_DIM})"
