"""Evaluation metrics (guide §2.5). Pure functions over a cached run result and a label file.

Requirement matching: each labelled requirement is matched one-to-one (greedy, best
score first) to an extracted requirement when similarity >= threshold.
Similarity = max(WRatio vs extracted text, WRatio vs its verbatim quote), computed on
CONTENT words (RFP boilerplate such as "the bidder must have a valid" removed), and 0
unless every number in the label also appears in that candidate text.
Why not plain ratio on full text (the guide's example): boilerplate dominates character
similarity. It scored "CMMI Level 5 certificate" vs "ISO 27001 certificate" at 82 (false
match), a real paraphrase at 79 (miss), and WRatio matched "office in New Delhi" to
unrelated requirements at 85.5. On content words, true pairs scored 89-100, non-matches <=65.
"""
import re

from rapidfuzz import fuzz

from rfp.nodes.verifier import norm

MATCH_THRESHOLD = 80
BOILERPLATE = set("""a an the of in on for to and or be is are must shall should will have has
having hold holds valid bidder bidders agency vendor company firm any all at least with by
from as per its their this that these those than""".split())


def _numbers(s: str) -> set[str]:
    return set(re.findall(r"\d+", s))


def content_words(s: str) -> str:
    return " ".join(w for w in re.findall(r"[a-z0-9]+", norm(s)) if w not in BOILERPLATE)


def similarity(label_text: str, req: dict) -> float:
    lab = content_words(label_text)
    best = 0.0
    for cand in (content_words(req["text"]), content_words(req["citation"]["exact_quote"])):
        if _numbers(lab) <= _numbers(cand):  # figures must agree, as in the citation verifier
            best = max(best, fuzz.WRatio(lab, cand))
    return best


def match_requirements(labelled: list[dict], extracted: list[dict],
                       threshold: float = MATCH_THRESHOLD) -> list[tuple[int, int, float]]:
    """[(label_idx, extracted_idx, score)] one-to-one matches at or above threshold."""
    pairs = sorted(((similarity(lab["text"], ex), i, j)
                    for i, lab in enumerate(labelled) for j, ex in enumerate(extracted)),
                   reverse=True)
    used_l, used_e, matches = set(), set(), []
    for score, i, j in pairs:
        if score < threshold:
            break
        if i not in used_l and j not in used_e:
            used_l.add(i); used_e.add(j)
            matches.append((i, j, round(score, 1)))
    return matches


def requirement_metrics(labelled, extracted, threshold=MATCH_THRESHOLD) -> dict:
    m = match_requirements(labelled, extracted, threshold)
    return {"labelled": len(labelled), "extracted": len(extracted), "matched": len(m),
            "recall": len(m) / len(labelled) if labelled else None,
            "precision": len(m) / len(extracted) if extracted else None,
            "matches": m}


def field_accuracy(label: dict, extraction: dict) -> dict:
    dl = (extraction.get("submission_deadline") or {}).get("value")
    val = (extraction.get("estimated_value_inr") or {}).get("value")
    exp_val = label.get("estimated_value_inr")
    return {
        "deadline_correct": None if label.get("deadline") is None else dl == label["deadline"],
        "value_correct": None if exp_val is None else (val is not None and abs(val - exp_val) < 1),
    }


def cost_latency(calls: list[dict]) -> dict:
    return {"llm_calls": len(calls),
            "input_tokens": sum(c["input_tokens"] or 0 for c in calls),
            "output_tokens": sum(c["output_tokens"] or 0 for c in calls),
            "latency_s": round(sum(c["latency_ms"] or 0 for c in calls) / 1000, 1),
            "cost_usd": round(sum(float(c["cost_usd"] or 0) for c in calls), 6)}


def time_saved(manual_minutes, pipeline_minutes, review_minutes):
    """manual − (pipeline + human review). None when the manual baseline is not measured."""
    if manual_minutes is None or review_minutes is None:
        return None
    return round(manual_minutes - (pipeline_minutes + review_minutes), 1)


def roi_inr(saved_minutes, cost_usd, hourly_rate_inr, usd_inr):
    """ASSUMPTION-based: hourly rate and FX rate are inputs, not facts."""
    if saved_minutes is None:
        return None
    return round(saved_minutes / 60 * hourly_rate_inr - cost_usd * usd_inr, 0)


def evaluate(label: dict, result: dict, assumptions: dict) -> dict:
    ex = result["extraction"] or {"requirements": []}
    req = requirement_metrics(label["requirements"], ex["requirements"])
    perf = cost_latency(result["calls"])
    saved = time_saved(label.get("manual_minutes"), perf["latency_s"] / 60,
                       label.get("review_minutes", assumptions.get("review_minutes")))
    rec = (result.get("score") or {}).get("recommendation")
    return {
        "name": label["name"], "status": result["status"], **req,
        **field_accuracy(label, ex),
        "citation_pass_rate": (result.get("redaction_stats") or {}).get("citation_pass_rate"),
        "recommendation": rec, "expected_recommendation": label.get("expected_recommendation"),
        "score_agrees": (None if label.get("expected_recommendation") is None
                         else rec == label["expected_recommendation"]),
        **perf, "time_saved_min": saved,
        "roi_inr": roi_inr(saved, perf["cost_usd"], assumptions["hourly_rate_inr"],
                           assumptions["usd_inr"]),
    }
