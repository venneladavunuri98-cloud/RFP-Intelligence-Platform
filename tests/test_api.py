"""FastAPI endpoints against the real graph (Postgres checkpoints, fake LLM)."""
import pytest
from fastapi.testclient import TestClient

from rfp import config
from rfp.api import main as api
from tests.conftest import make_pdf
from tests.test_graph import PAGE1, PAGE2, fake  # noqa: F401  (fixture)


@pytest.fixture
def client(pg_storage, fake, tmp_path, monkeypatch):  # noqa: F811
    fake()
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path / "uploads")
    sent = []
    monkeypatch.setattr(config, "N8N_WEBHOOK_URL", "http://n8n.test/webhook/rfp")

    class Resp:
        def raise_for_status(self):
            pass
    monkeypatch.setattr(api.requests, "post", lambda url, json, timeout: sent.append(json) or Resp())
    with TestClient(api.create_app()) as c:
        c.sent = sent
        c.pdf = make_pdf(tmp_path / "rfp.pdf", [PAGE1, PAGE2])
        yield c


def upload(c, path=None, data=None):
    body = data if data is not None else open(path or c.pdf, "rb").read()
    return c.post("/evaluate", files={"file": ("tender.pdf", body, "application/pdf")})


def test_evaluate_runs_until_review_and_lists_it(client):
    r = upload(client)
    assert r.status_code == 200 and r.json()["awaiting_review"] is True
    run_id = r.json()["run_id"]
    assert run_id in [x["run_id"] for x in client.get("/runs").json()]
    run = client.get(f"/runs/{run_id}").json()
    assert run["awaiting_review"] and run["review_payload"]["score"]["recommendation"]


def test_override_without_reason_is_rejected_at_the_api(client):
    run_id = upload(client).json()["run_id"]
    r = client.post(f"/runs/{run_id}/review", json={"action": "override", "reviewer": "me",
                                                    "reason": "  ", "final_recommendation": "no_go"})
    assert r.status_code == 422 and "reason" in r.json()["detail"]
    assert client.get(f"/runs/{run_id}").json()["awaiting_review"] is True  # graph untouched


@pytest.mark.parametrize("body", [{"action": "approve", "reviewer": ""},
                                  {"action": "delete", "reviewer": "me"},
                                  {"action": "override", "reviewer": "me", "reason": "x"}])
def test_invalid_decisions_rejected(client, body):
    run_id = upload(client).json()["run_id"]
    assert client.post(f"/runs/{run_id}/review", json=body).status_code == 422


def test_approve_finalizes_notifies_and_leaves_queue(client):
    run_id = upload(client).json()["run_id"]
    r = client.post(f"/runs/{run_id}/review", json={"action": "approve", "reviewer": "Vennela"})
    assert r.status_code == 200 and r.json()["status"] == "final"
    assert r.json()["notification"] == "sent"
    [event] = client.sent
    assert event["run_id"] == run_id and event["review"]["reviewer"] == "Vennela"
    assert event["review"]["final_recommendation"] == event["score"]["recommendation"]
    assert run_id not in [x["run_id"] for x in client.get("/runs").json()]
    again = client.post(f"/runs/{run_id}/review", json={"action": "approve", "reviewer": "x"})
    assert again.status_code == 409  # cannot decide twice


def test_override_with_reason_carries_reviewer_notes(client):
    run_id = upload(client).json()["run_id"]
    r = client.post(f"/runs/{run_id}/review", json={
        "action": "override", "reviewer": "Vennela", "reason": "Unlimited liability",
        "final_recommendation": "no_go"})
    assert r.json()["review"]["final_recommendation"] == "no_go"
    assert client.sent[0]["review"]["reason"] == "Unlimited liability"


def test_notification_failure_does_not_undo_decision(client, monkeypatch):
    import requests

    def boom(*a, **k):
        raise requests.ConnectionError("n8n down")
    monkeypatch.setattr(api.requests, "post", boom)
    run_id = upload(client).json()["run_id"]
    r = client.post(f"/runs/{run_id}/review", json={"action": "approve", "reviewer": "me"})
    assert r.status_code == 200 and r.json()["status"] == "final"
    assert r.json()["notification"].startswith("notification failed")


def test_page_png(client):
    run_id = upload(client).json()["run_id"]
    r = client.get(f"/runs/{run_id}/page/2")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert r.content.startswith(b"\x89PNG")
    assert client.get(f"/runs/{run_id}/page/9").status_code == 404


def test_rejects_non_pdf_and_oversize(client, monkeypatch):
    assert upload(client, data=b"<html>not a pdf</html>").status_code == 415
    monkeypatch.setattr(config, "MAX_UPLOAD_MB", 0)
    assert upload(client).status_code == 413


def test_scanned_pdf_reaches_review_as_error(client, tmp_path):
    r = upload(client, path=make_pdf(tmp_path / "scan.pdf", ["", ""]))
    run = client.get(f"/runs/{r.json()['run_id']}").json()
    assert r.json()["awaiting_review"] and run["review_payload"]["status"] == "error"
    assert run["review_payload"]["score"] is None


def test_unknown_run_is_404(client):
    assert client.get("/runs/nope").status_code == 404
    assert client.post("/runs/nope/review", json={"action": "approve", "reviewer": "me"}).status_code == 404


def test_report_available_after_final_and_safe(client):
    run_id = upload(client).json()["run_id"]
    assert client.get(f"/runs/{run_id}/report").status_code == 404      # not final yet
    client.post(f"/runs/{run_id}/review", json={"action": "approve", "reviewer": "Vennela"})
    r = client.get(f"/runs/{run_id}/report")
    assert r.status_code == 200 and "Final recommendation" in r.text and "Vennela" in r.text
    assert "default-src 'none'" in r.headers["content-security-policy"]
    assert client.get("/runs/..%2F..%2Fetc%2Fpasswd/report").status_code == 404


def test_api_starts_even_if_capability_ingest_fails(pg_storage, monkeypatch):
    # Regression: on a fresh DB the startup ingest called the (blocked/unavailable)
    # embedding API and the whole API failed to start.
    from rfp import db
    with db.connect(pg_storage) as conn:
        conn.execute("TRUNCATE capabilities")
    with TestClient(api.create_app()) as c:
        assert c.get("/health").json() == {"ok": True}
