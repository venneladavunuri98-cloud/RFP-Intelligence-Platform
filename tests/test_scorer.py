import copy
import json
from datetime import date

import pytest

from rfp.nodes.scorer import load_rubric, score, score_node

TODAY = date(2026, 10, 15)
RUBRIC = load_rubric()
C = {"page_number": 1, "exact_quote": "placeholder quote text"}


def req(id, cat, grounded=True, mandatory=True):
    return {"id": id, "text": id, "category": cat, "mandatory": mandatory,
            "citation": C, "grounded": grounded}


def make_state(**over):
    state = {
        "extraction": {
            "title": {"value": "T", "citation": C, "grounded": True},
            "submission_deadline": {"value": "2026-11-30", "citation": C, "grounded": True},
            "estimated_value_inr": {"value": 25_000_000, "citation": C, "grounded": True},
            "requirements": [req("R1", "technical"), req("R2", "technical"),
                             req("R3", "eligibility"), req("R4", "legal_risk")],
        },
        "compliance": [{"requirement_id": "R1", "status": "covered", "evidence": "e", "reasoning": "r"},
                       {"requirement_id": "R2", "status": "partial", "evidence": "e", "reasoning": "r"},
                       {"requirement_id": "R3", "status": "covered", "evidence": "e", "reasoning": "r"}],
        "risk_flags": [],
    }
    state.update(over)
    return state


def test_hand_computed_total():
    s = score(make_state(), RUBRIC, today=TODAY)
    assert s["breakdown"] == {"technical_fit": 0.75, "eligibility_fit": 1.0,
                              "timeline_feasibility": 1.0, "commercial_fit": 1.0, "legal_risk": 0.75}
    # (35*.75 + 25*1 + 15*1 + 10*1 + 15*.75) / 100 * 100
    assert s["total"] == 87.5 and s["recommendation"] == "go" and s["rubric_version"] == 1


def test_same_input_identical_output():
    a = score(make_state(), RUBRIC, today=TODAY)
    b = score(make_state(), RUBRIC, today=TODAY)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_score_does_not_mutate_state():
    st = make_state()
    before = copy.deepcopy(st)
    score(st, RUBRIC, today=TODAY)
    assert st == before


def test_ungrounded_requirement_excluded_and_caps_go():
    st = make_state()
    st["extraction"]["requirements"].append(req("R5", "technical", grounded=False))
    st["compliance"].append({"requirement_id": "R5", "status": "gap", "evidence": "", "reasoning": ""})
    s = score(st, RUBRIC, today=TODAY)
    assert s["breakdown"]["technical_fit"] == 0.75  # R5's gap ignored
    assert "1 ungrounded requirement(s) excluded from scoring" in s["reasons"]
    assert s["recommendation"] == "review"  # capped from go


def test_missing_deadline_is_neutral():
    st = make_state()
    st["extraction"]["submission_deadline"] = None
    s = score(st, RUBRIC, today=TODAY)
    assert s["breakdown"]["timeline_feasibility"] == RUBRIC["unknown_score"]
    assert "deadline missing/ungrounded" in s["reasons"]


def test_ungrounded_deadline_is_neutral():
    st = make_state()
    st["extraction"]["submission_deadline"]["grounded"] = False
    assert score(st, RUBRIC, today=TODAY)["breakdown"]["timeline_feasibility"] == 0.5


@pytest.mark.parametrize("value", ["05/11/2026", "within 30 days of publication"])
def test_non_iso_deadline_is_neutral_not_a_crash(value):
    st = make_state()
    st["extraction"]["submission_deadline"]["value"] = value
    s = score(st, RUBRIC, today=TODAY)
    assert s["breakdown"]["timeline_feasibility"] == 0.5
    assert any("not an ISO date" in r for r in s["reasons"])


@pytest.mark.parametrize("deadline,expected", [
    ("2026-10-10", 0.0),   # already passed
    ("2026-10-22", 0.0),   # 7 days: zero-score boundary
    ("2026-10-29", 0.5),   # 14 days: halfway
    ("2026-11-05", 1.0),   # 21 days: full-score boundary
])
def test_timeline_scaling(deadline, expected):
    st = make_state()
    st["extraction"]["submission_deadline"]["value"] = deadline
    assert score(st, RUBRIC, today=TODAY)["breakdown"]["timeline_feasibility"] == expected


@pytest.mark.parametrize("value,expected", [(1_000_000, 0.0), (12_500_000, 0.5), (50_000_000, 1.0)])
def test_commercial_scaling(value, expected):
    st = make_state()
    st["extraction"]["estimated_value_inr"]["value"] = value
    assert score(st, RUBRIC, today=TODAY)["breakdown"]["commercial_fit"] == expected


def test_missing_compliance_is_neutral_not_covered():
    s = score(make_state(compliance=[]), RUBRIC, today=TODAY)
    assert s["breakdown"]["technical_fit"] == 0.5
    assert s["breakdown"]["eligibility_fit"] == 0.5
    assert any("without compliance evidence" in r for r in s["reasons"])


def test_gap_scores_zero():
    st = make_state()
    st["compliance"] = [{"requirement_id": i, "status": "gap", "evidence": "", "reasoning": ""}
                        for i in ("R1", "R2", "R3")]
    s = score(st, RUBRIC, today=TODAY)
    assert s["breakdown"]["technical_fit"] == 0.0 and s["breakdown"]["eligibility_fit"] == 0.0


def test_no_mandatory_requirements_in_category_is_neutral():
    st = make_state()
    st["extraction"]["requirements"] = [req("R1", "technical", mandatory=False)]
    s = score(st, RUBRIC, today=TODAY)
    assert s["breakdown"]["technical_fit"] == 0.5 and s["breakdown"]["legal_risk"] == 1.0


def test_legal_risk_penalty_per_clause():
    st = make_state()
    st["extraction"]["requirements"] += [req(f"L{i}", "legal_risk") for i in range(4)]
    assert score(st, RUBRIC, today=TODAY)["breakdown"]["legal_risk"] == 0.0  # 5 clauses, clamped


def test_risk_flag_caps_go_at_review():
    s = score(make_state(risk_flags=["page 2: matched /system prompt/"]), RUBRIC, today=TODAY)
    assert s["total"] == 87.5 and s["recommendation"] == "review"
    assert any("capped" in r for r in s["reasons"])


def test_risk_flag_never_upgrades_no_go():
    st = make_state(risk_flags=["x"], compliance=[])
    st["extraction"]["submission_deadline"]["value"] = "2026-10-16"
    st["extraction"]["estimated_value_inr"]["value"] = 1
    s = score(st, RUBRIC, today=TODAY)
    assert s["recommendation"] == "no_go"
    assert not any("capped" in r for r in s["reasons"])  # only reported when it applies


def test_thresholds():
    st = make_state(compliance=[])  # tech .5, elig .5 -> 66.2 -> review
    s = score(st, RUBRIC, today=TODAY)
    assert 50 <= s["total"] < 70 and s["recommendation"] == "review"


def test_rubric_with_wrong_weights_is_rejected(tmp_path):
    bad = tmp_path / "rubric.yaml"
    bad.write_text("version: 1\nweights: {technical_fit: 100}\n")
    with pytest.raises(ValueError):
        load_rubric(bad)


def test_score_node_uses_rubric_file():
    out = score_node(make_state())
    assert out["score"]["rubric_version"] == 1 and set(out["score"]["breakdown"]) == set(RUBRIC["weights"])
