"""Compliance agent with a deterministic fake embedder/judge and a real pgvector DB."""
import pytest

from rfp import capabilities, config, llm
from rfp.nodes import compliance
from rfp.nodes.scorer import load_rubric, score
from rfp.schemas import ComplianceFlag, ComplianceResult

KEYWORDS = ["iso", "cmmi", "office", "delhi", "uptime", "turnover", "encrypt", "liability",
            "subcontract", "blacklist", "cert-in", "gigw"]


def fake_embed(texts):
    """Bag-of-keywords vectors: retrieval is deterministic and explainable in tests."""
    vecs = []
    for t in texts:
        v = [float(t.lower().count(k)) for k in KEYWORDS]
        v += [0.0] * (config.EMBED_DIM - len(v) - 1) + [0.01]  # never a zero vector
        vecs.append(v)
    return vecs


@pytest.fixture
def store(pg_uri, monkeypatch):
    monkeypatch.setattr(config, "DB_URI", pg_uri)
    monkeypatch.setattr(config, "RETRY_BACKOFF_S", 0)
    monkeypatch.setattr(llm, "embed_queries", fake_embed)
    capabilities.ingest(uri=pg_uri, embed=fake_embed)
    return pg_uri


class FakeJudge:
    """Each scripted step is a list of ComplianceFlags (one batched response) or an Exception."""
    def __init__(self, *steps):
        self.steps, self.prompts = list(steps), []

    def invoke(self, messages):
        self.prompts.append(messages[1][1])
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return {"raw": None, "parsed": ComplianceResult(flags=step), "parsing_error": None}


def use_judge(monkeypatch, *steps):
    j = FakeJudge(*steps)
    monkeypatch.setattr(compliance, "get_judge", lambda: j)
    return j


def req(id, text, grounded=True, mandatory=True, cat="eligibility"):
    return {"id": id, "text": text, "category": cat, "mandatory": mandatory, "grounded": grounded,
            "citation": {"page_number": 1, "exact_quote": "placeholder quote"}}


def state(*reqs):
    return {"run_id": "c1", "extraction": {"requirements": list(reqs)}}


# ---- ingestion ----

def test_split_sections_reads_all_mock_sections():
    secs = capabilities.split_sections(open(config.ROOT / "data/capabilities.md").read())
    assert len(secs) == 20
    assert secs[0][0] == "Legal entity and registration"
    assert all(body for _, body in secs)


def test_ingest_is_idempotent(store):
    from rfp import db
    capabilities.ingest(uri=store, embed=fake_embed)
    with db.connect(store) as conn:
        assert conn.execute("SELECT count(*) FROM capabilities").fetchone()[0] == 20


def test_ingest_rejects_wrong_dimension(pg_uri):
    with pytest.raises(ValueError):
        capabilities.ingest(uri=pg_uri, embed=lambda texts: [[1.0, 2.0]] * len(texts))


def test_retrieval_ranks_relevant_section_first(store):
    from rfp import db
    with db.connect(store) as conn:
        top = compliance.retrieve(conn, fake_embed(["CMMI level 5 appraisal"])[0])
    assert len(top) == 3 and top[0]["section"] == "Quality certifications"
    assert top[0]["distance"] <= top[1]["distance"] <= top[2]["distance"]


# ---- judging ----

def flag(rid, status, evidence="", reasoning="r"):
    return ComplianceFlag(requirement_id=rid, status=status, evidence=evidence, reasoning=reasoning)


def test_one_batched_call_judges_all_and_verifies_evidence(store, monkeypatch):
    judge = use_judge(monkeypatch, [
        flag("R1", "gap", "We do NOT hold a CMMI Level 5 appraisal."),
        flag("R2", "covered", "ISO/IEC 27001:2022 certified Information Security Management System"),
        flag("R3", "covered", "We guarantee 99.99% uptime on all contracts."),   # invented
    ])
    out = compliance.check(state(req("R1", "Bidder must hold CMMI Level 5"),
                                 req("R2", "Bidder must hold ISO 27001 certification"),
                                 req("R3", "Bidder must provide 99.99% uptime", cat="technical")))
    f1, f2, f3 = out["compliance"]
    assert len(judge.prompts) == 1                       # one LLM call for all requirements
    assert judge.prompts[0].count("<item>") == 3
    assert f1["retrieved"][0]["section"] == "Quality certifications"
    assert f2["evidence_grounded"] is True               # quote found in R2's own passages
    assert f3["evidence_grounded"] is False              # invented evidence is caught
    assert "errors" not in out


def test_evidence_must_come_from_that_requirements_own_passages(store, monkeypatch):
    # R2 (ISO) cites the CMMI sentence, which was only retrieved for R1.
    use_judge(monkeypatch, [flag("R1", "gap", "We do NOT hold a CMMI Level 5 appraisal."),
                            flag("R2", "covered", "We do NOT hold a CMMI Level 5 appraisal.")])
    out = compliance.check(state(req("R1", "CMMI Level 5 cmmi"), req("R2", "uptime uptime")))
    assert out["compliance"][1]["evidence_grounded"] is False


def test_unknown_ids_dropped_and_missing_verdicts_reported(store, monkeypatch):
    use_judge(monkeypatch, [flag("R1", "gap"), flag("R1", "covered"), flag("R99", "covered")])
    out = compliance.check(state(req("R1", "ISO 27001"), req("R2", "CMMI")))
    assert [(f["requirement_id"], f["status"]) for f in out["compliance"]] == [("R1", "gap")]
    assert "compliance R2: no verdict returned" in out["errors"]


def test_only_grounded_mandatory_requirements_are_judged(store, monkeypatch):
    judge = use_judge(monkeypatch, [flag("R1", "gap")])
    out = compliance.check(state(req("R1", "ISO 27001"), req("R2", "x", grounded=False),
                                 req("R3", "y", mandatory=False)))
    assert judge.prompts[0].count("<item>") == 1
    assert [f["requirement_id"] for f in out["compliance"]] == ["R1"]


def test_judge_failure_leaves_requirements_unflagged_and_reported(store, monkeypatch):
    use_judge(monkeypatch, *[RuntimeError("503")] * 3)
    out = compliance.check(state(req("R1", "ISO 27001")))
    assert out["compliance"] == [] and "compliance: judge failed" in out["errors"][0]


def test_daily_quota_error_is_not_retried(store, monkeypatch):
    judge = use_judge(monkeypatch, RuntimeError(
        "429 RESOURCE_EXHAUSTED quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier"))
    out = compliance.check(state(req("R1", "ISO 27001")))
    assert len(judge.prompts) == 1 and out["compliance"] == []
    assert "daily quota exhausted" in out["errors"][0]


def test_database_down_fails_closed(monkeypatch):
    monkeypatch.setattr(config, "DB_URI", "postgresql://rfp:rfp@127.0.0.1:1/nope")
    monkeypatch.setattr(llm, "embed_queries", fake_embed)
    out = compliance.check(state(req("R1", "ISO 27001")))
    assert out["compliance"] == [] and "compliance unavailable" in out["errors"][0]


def test_unverified_evidence_scores_neutral_not_covered():
    st = {"extraction": {"requirements": [req("R1", "uptime", cat="technical")],
                         "submission_deadline": None, "estimated_value_inr": None},
          "compliance": [{"requirement_id": "R1", "status": "covered", "evidence": "invented",
                          "reasoning": "", "evidence_grounded": False}]}
    s = score(st, load_rubric())
    assert s["breakdown"]["technical_fit"] == 0.5
    assert any("unverified evidence" in r for r in s["reasons"])


def test_evidence_with_section_title_prefix_verifies():
    passages = [{"section": "Years of experience", "content": "The company has 13 years of experience."}]
    assert compliance.evidence_in_passages(
        "Years of experience: The company has 13 years of experience.", passages)
    assert not compliance.evidence_in_passages("The company has 30 years of experience.", passages)
