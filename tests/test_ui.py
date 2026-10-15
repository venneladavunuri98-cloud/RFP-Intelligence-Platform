"""Streamlit console driven by AppTest against a real uvicorn API (fake LLM, test DB)."""
import socket
import threading
import time

import pytest
import requests
import uvicorn
from streamlit.testing.v1 import AppTest

from rfp import config
from rfp.api import main as api
from tests.conftest import make_pdf
from tests.test_graph import PAGE1, PAGE2, fake  # noqa: F401  (fixture)

UI = str(config.ROOT / "ui" / "app.py")


@pytest.fixture
def live_api(pg_storage, fake, tmp_path, monkeypatch):  # noqa: F811
    fake()
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(config, "N8N_WEBHOOK_URL", "")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(api.create_app(), host="127.0.0.1", port=port,
                                           log_level="warning"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            requests.get(f"{url}/health", timeout=1)
            break
        except requests.RequestException:
            time.sleep(0.05)
    monkeypatch.setenv("API_URL", url)
    pdf = make_pdf(tmp_path / "rfp.pdf", [PAGE1, PAGE2])
    run_id = requests.post(f"{url}/evaluate", files={"file": ("t.pdf", open(pdf, "rb").read(),
                                                              "application/pdf")}).json()["run_id"]
    yield url, run_id
    server.should_exit = True
    t.join(timeout=5)


def open_detail(run_id):
    at = AppTest.from_file(UI, default_timeout=30)
    at.session_state["screen"] = "Run detail"
    at.session_state["run_id"] = run_id
    return at.run()


def test_queue_lists_the_run(live_api):
    _, run_id = live_api
    at = AppTest.from_file(UI, default_timeout=30).run()
    assert not at.exception
    assert any(run_id in m.value for m in at.markdown)


def test_detail_shows_score_grounding_and_quotes(live_api):
    _, run_id = live_api
    at = open_detail(run_id)
    assert not at.exception
    assert at.metric[0].label == "Score"
    labels = [e.label for e in at.expander]
    assert any(l.startswith("✅ R1") for l in labels)
    assert any("unlimited liability" in m.value for m in at.markdown)


def test_override_submit_disabled_until_reason(live_api):
    url, run_id = live_api
    at = open_detail(run_id)
    at.text_input(key="reviewer").input("Vennela").run()
    at.radio(key="action").set_value("override").run()
    assert at.button(key="submit").disabled                 # no reason yet
    at.text_area(key="reason").input("Unlimited liability is unacceptable").run()
    assert not at.button(key="submit").disabled
    at.button(key="submit").click().run()
    assert not at.exception
    run = requests.get(f"{url}/runs/{run_id}").json()
    assert run["status"] == "final" and run["review"]["reason"] == "Unlimited liability is unacceptable"


def test_open_button_navigates_to_detail(live_api):
    # Regression: assigning the radio's session_state key after it was drawn crashed the app.
    _, run_id = live_api
    at = AppTest.from_file(UI, default_timeout=30).run()
    at.button(key=f"open-{run_id}").click().run()
    assert not at.exception
    assert at.session_state["screen"] == "Run detail" and at.metric[0].label == "Score"


def test_final_run_still_shows_requirements(live_api):
    # Regression: after finalization the interrupt payload is gone and the console
    # showed "Requirements (0)" even though the run had requirements.
    url, run_id = live_api
    requests.post(f"{url}/runs/{run_id}/review", json={"action": "approve", "reviewer": "Vennela"})
    at = open_detail(run_id)
    assert not at.exception
    assert any("FINAL" in s.value for s in at.success)
    assert any(h.value.startswith("Requirements (3)") for h in at.subheader)
