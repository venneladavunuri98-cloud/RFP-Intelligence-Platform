"""Terminal runner driven with scripted answers instead of a keyboard."""
import os

from rfp import config
from rfp.run import run
from tests.conftest import make_pdf
from tests.test_graph import PAGE1, PAGE2, fake  # noqa: F401  (fixture)


def scripted(*answers):
    it = iter(answers)
    return lambda prompt: next(it)


def test_terminal_flow_rejects_reasonless_override_then_approves(fake, tmp_path):  # noqa: F811
    fake()
    pdf = make_pdf(tmp_path / "rfp.pdf", [PAGE1, PAGE2])
    printed = []
    res = run(pdf, "cli-1", ask=scripted("me", "override", "no_go", "",   # no reason -> rejected
                                         "me", "approve"),
              echo=printed.append)
    text = "\n".join(printed)
    assert "SCORE" in text and "quote:" in text
    assert "REJECTED DECISION: override requires a written reason" in text
    assert res["status"] == "final" and res["review"]["action"] == "approve"
    assert "=== FINAL:" in printed[-2]


def test_dotenv_loader_does_not_override_real_env(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nRFP_TEST_A=from_file\nRFP_TEST_B='quoted'\n")
    monkeypatch.setenv("RFP_TEST_A", "from_env")
    monkeypatch.delenv("RFP_TEST_B", raising=False)
    config.load_dotenv(env)
    assert os.environ["RFP_TEST_A"] == "from_env" and os.environ["RFP_TEST_B"] == "quoted"
    monkeypatch.delenv("RFP_TEST_B")


def test_render_shows_compliance_evidence():
    from rfp.run import render_draft
    text = render_draft({"status": "ok", "compliance": [{
        "requirement_id": "R1", "status": "gap", "reasoning": "Only CMMI Level 3.",
        "evidence": "We do NOT hold a CMMI Level 5 appraisal.", "evidence_grounded": True,
        "retrieved": [{"section": "Quality certifications", "content": "...", "distance": 0.21}]}]})
    assert "R1: GAP - Only CMMI Level 3." in text
    assert 'OK evidence: "We do NOT hold a CMMI Level 5 appraisal."' in text
    assert "Quality certifications (0.21)" in text
