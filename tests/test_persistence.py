"""Stage 2 persistence: Postgres checkpoints, telemetry table, audit rows, kill-and-resume."""
import os
import signal
import subprocess
import sys
import time
import uuid

import psycopg
import pytest
from langchain_core.messages import AIMessage
from langgraph.types import Command

from rfp import config, db, telemetry
from rfp.graph import open_graph
from tests.conftest import make_pdf
from tests.test_graph import PAGE1, PAGE2, fake  # noqa: F401  (fixture)

ROOT = config.ROOT


def q(uri, sql, *args):
    with psycopg.connect(uri) as conn:
        return conn.execute(sql, args).fetchall()


def test_telemetry_goes_to_audit_table(pg_storage):
    run_id = f"tel-{uuid.uuid4().hex[:6]}"
    msg = AIMessage(content="", usage_metadata={"input_tokens": 10, "output_tokens": 5,
                                                "total_tokens": 15})
    telemetry.log_call(run_id, "extract", "m", msg, time.perf_counter(), attempt=2, ok=True)
    rows = q(pg_storage, "SELECT node, input_tokens, output_tokens, attempt, ok, error "
                         "FROM audit_llm_calls WHERE run_id = %s", run_id)
    assert rows == [("extract", 10, 5, 2, True, None)]
    assert not config.TELEMETRY_LOG.exists()  # no JSONL when the DB write succeeds


def test_telemetry_falls_back_to_jsonl_when_db_down(pg_storage, monkeypatch):
    monkeypatch.setattr(config, "DB_URI", "postgresql://rfp:rfp@127.0.0.1:1/nope")
    rec = telemetry.log_call("tel-down", "extract", "m", None, time.perf_counter())
    assert "db_error" in rec and "tel-down" in config.TELEMETRY_LOG.read_text()


def test_checkpoint_survives_new_pool_and_writes_audit_rows(pg_storage, fake, tmp_path):  # noqa: F811
    fake()
    run_id = f"ckpt-{uuid.uuid4().hex[:6]}"
    cfg = {"configurable": {"thread_id": run_id}}
    pdf = make_pdf(tmp_path / "rfp.pdf", [PAGE1, PAGE2])
    with open_graph() as graph:                      # "process 1"
        graph.invoke({"run_id": run_id, "pdf_path": pdf}, cfg)
    assert q(pg_storage, "SELECT status FROM runs WHERE run_id=%s", run_id) == [("processing",)]

    with open_graph() as graph:                      # "process 2": new pool, new graph
        st = graph.get_state(cfg)
        assert st.next == ("human_review",)
        assert list(st.values["pages"]) == [1, 2]    # int page keys survive Postgres serde
        res = graph.invoke(Command(resume={"action": "override", "reviewer": "me",
                                           "reason": "unlimited liability",
                                           "final_recommendation": "no_go"}), cfg)
    assert res["status"] == "final"
    [(action, reason, reviewer, ai, final, score)] = q(
        pg_storage, "SELECT action, reason, reviewer, ai_recommendation, final_recommendation, "
                    "score FROM reviews WHERE run_id=%s", run_id)
    assert (action, reason, reviewer, final) == ("override", "unlimited liability", "me", "no_go")
    assert ai == res["score"]["recommendation"] and score["total"] == res["score"]["total"]
    assert q(pg_storage, "SELECT status FROM runs WHERE run_id=%s", run_id) == [("final",)]


def test_write_review_is_idempotent(pg_storage):
    review = {"action": "approve", "reviewer": "me", "final_recommendation": "go",
              "decided_at": "2026-10-15T00:00:00+00:00"}
    db.write_review("idem-1", review, {"total": 1})
    db.write_review("idem-1", {**review, "final_recommendation": "review"}, {"total": 1})
    assert q(pg_storage, "SELECT final_recommendation FROM reviews WHERE run_id='idem-1'") == \
        [("review",)]


@pytest.mark.skipif(sys.platform == "win32", reason="SIGKILL does not exist on Windows")
def test_kill_dash_nine_at_review_prompt_then_resume_in_new_process(pg_storage, tmp_path):
    """Real OS processes: the first is SIGKILLed while waiting for the reviewer."""
    run_id = f"kill-{uuid.uuid4().hex[:6]}"
    pdf = make_pdf(tmp_path / "scan.pdf", ["", ""])   # error path: no LLM call needed
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "RFP_STORAGE": "postgres",
           "DB_URI": pg_storage, "TELEMETRY_LOG": str(tmp_path / "t.jsonl")}
    cmd = [sys.executable, "-m", "rfp.run"]

    p1 = subprocess.Popen([*cmd, pdf, "--thread-id", run_id], cwd=ROOT, env=env,
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    deadline = time.time() + 60
    while time.time() < deadline:                       # wait until it is parked at the prompt
        rows = q(pg_storage, "SELECT status FROM runs WHERE run_id=%s", run_id)
        if rows == [("awaiting_review",)]:
            break
        time.sleep(0.2)
    else:
        p1.kill()
        raise AssertionError("first process never reached the review prompt")
    time.sleep(0.5)
    p1.send_signal(signal.SIGKILL)
    assert p1.wait(timeout=10) == -signal.SIGKILL

    p2 = subprocess.run([*cmd, "--thread-id", run_id], cwd=ROOT, env=env, input=b"me\napprove\n",
                        capture_output=True, timeout=60)
    out = p2.stdout.decode()
    assert p2.returncode == 0, out + p2.stderr.decode()
    assert f"Resuming run {run_id} from its checkpoint." in out
    assert "=== FINAL: MANUAL_REVIEW_REQUIRED ===" in out
    assert q(pg_storage, "SELECT action, final_recommendation FROM reviews WHERE run_id=%s",
             run_id) == [("approve", "manual_review_required")]
