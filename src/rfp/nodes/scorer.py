"""Deterministic scorer (guide §1.7): pure Python + rubric.yaml, no LLM.

Only grounded fields are used; ungrounded claims are excluded and reported.
Passing `today` makes the score reproducible.
"""
from datetime import date

import yaml
from dateutil import parser as dtp

from .. import config
from ..schemas import RFPScore

PARTS = {"technical_fit", "eligibility_fit", "timeline_feasibility", "commercial_fit", "legal_risk"}


def load_rubric(path=None) -> dict:
    with open(path or config.RUBRIC_PATH) as f:
        rubric = yaml.safe_load(f)
    if set(rubric["weights"]) != PARTS:
        raise ValueError(f"rubric weights must be exactly {sorted(PARTS)}")
    return rubric


def _clamp(x):
    return max(0.0, min(1.0, x))


def score(state, rubric, today=None):
    today = today or date.today()
    ex, flags = state["extraction"], {f["requirement_id"]: f for f in state.get("compliance", [])}
    unk, reasons, parts = rubric["unknown_score"], [], {}
    reqs = [r for r in ex["requirements"] if r["grounded"]]
    dropped = len(ex["requirements"]) - len(reqs)
    if dropped:
        reasons.append(f"{dropped} ungrounded requirement(s) excluded from scoring")

    def fit(cat):
        ids = [r["id"] for r in reqs if r["mandatory"] and r["category"] == cat]
        if not ids:
            reasons.append(f"{cat}: no grounded mandatory requirements, neutral score")
            return unk
        c = rubric["compliance"]
        per_req = {"covered": 1.0, "partial": 1 - c["partial_penalty"], "gap": 1 - c["gap_penalty"]}
        # No compliance flag = unknown (neutral), never silently "covered".
        missing = [i for i in ids if i not in flags]
        if missing:
            reasons.append(f"{cat}: {len(missing)} requirement(s) without compliance evidence, "
                           f"scored neutral")
        # Covered/partial claims whose evidence Python could not find in the
        # retrieved passages are treated like ungrounded data: neutral.
        unverified = [i for i in ids if i in flags and not flags[i].get("evidence_grounded", True)]
        if unverified:
            reasons.append(f"{cat}: {len(unverified)} compliance verdict(s) with unverified "
                           f"evidence, scored neutral")
        return _clamp(sum(unk if i not in flags or i in unverified
                          else per_req.get(flags[i]["status"], unk) for i in ids) / len(ids))

    parts["technical_fit"] = fit("technical")
    parts["eligibility_fit"] = fit("eligibility")

    dl = ex.get("submission_deadline")
    deadline = None
    if dl and dl["grounded"]:
        try:  # strict ISO only: "05/11/2026" is ambiguous (dateutil reads it as 11 May)
            deadline = dtp.isoparse(dl["value"]).date()
        except (ValueError, OverflowError):
            reasons.append(f"deadline {dl['value']!r} is not an ISO date, neutral score")
    else:
        reasons.append("deadline missing/ungrounded")
    if deadline:
        days = (deadline - today).days
        t = rubric["timeline"]
        parts["timeline_feasibility"] = _clamp((days - t["min_days_zero_score"]) /
                                               (t["min_days_full_score"] - t["min_days_zero_score"]))
    else:
        parts["timeline_feasibility"] = unk

    v = ex.get("estimated_value_inr")
    if v and v["grounded"]:
        c = rubric["commercial"]
        parts["commercial_fit"] = _clamp((v["value"] - c["min_value_inr"]) /
                                         (c["full_score_value_inr"] - c["min_value_inr"]))
    else:
        parts["commercial_fit"] = unk
        reasons.append("contract value missing/ungrounded")

    risky = [r for r in reqs if r["category"] == "legal_risk"]
    parts["legal_risk"] = _clamp(1 - rubric["legal"]["penalty_per_flagged_clause"] * len(risky))
    if risky:
        reasons.append(f"{len(risky)} legal-risk clause(s) flagged")

    w = rubric["weights"]
    total = round(100 * sum(w[k] * parts[k] for k in w) / sum(w.values()), 1)
    th = rubric["thresholds"]
    rec = "go" if total >= th["go"] else "review" if total >= th["review"] else "no_go"
    if (state.get("risk_flags") or dropped) and rec == "go":
        rec = "review"
        reasons.append("recommendation capped at 'review' due to risk flags or ungrounded claims")
    result = RFPScore(total=total, breakdown={k: round(v, 3) for k, v in parts.items()},
                      recommendation=rec, reasons=reasons, rubric_version=rubric["version"])
    return result.model_dump()


def score_node(state):
    """Graph node: thin wrapper that loads rubric.yaml and calls score()."""
    return {"score": score(state, load_rubric())}
