from eval.metrics import (cost_latency, evaluate, field_accuracy, match_requirements,
                          requirement_metrics, roi_inr, time_saved)


def ex(text, quote="placeholder verbatim quote"):
    return {"text": text, "citation": {"page_number": 1, "exact_quote": quote}}


LABELLED = [{"text": "Bidder must hold a valid CMMI Level 5 certificate"},
            {"text": "Average turnover above Rs. 50 lakh over five years"},
            {"text": "Must have an office in New Delhi"}]
EXTRACTED = [ex("The agency must hold a valid CMMI Level 5 certificate"),
             ex("Average turnover must exceed Rs. 50 lakh over five financial years"),
             ex("Website must be GIGW compliant"),
             ex("Bidder must hold a valid CMMI Level 5 certificate")]  # duplicate of label 0


def test_one_to_one_matching_and_precision_recall():
    m = requirement_metrics(LABELLED, EXTRACTED)
    assert m["matched"] == 2 and m["recall"] == 2 / 3 and m["precision"] == 2 / 4
    assert sorted(i for i, _, _ in m["matches"]) == [0, 1]
    assert len({j for _, j, _ in m["matches"]}) == 2  # no extracted item used twice


def test_match_can_use_verbatim_quote():
    extracted = [ex("CMMI requirement", quote="Bidder must hold a valid CMMI Level 5 certificate")]
    assert match_requirements(LABELLED[:1], extracted) == [(0, 0, 100.0)]


def test_unrelated_text_does_not_match():
    assert match_requirements(LABELLED[2:], [ex("Payment is quarterly in arrears")]) == []


def test_field_accuracy():
    extraction = {"submission_deadline": {"value": "2026-02-02"},
                  "estimated_value_inr": {"value": 7000000.0}}
    assert field_accuracy({"deadline": "2026-02-02", "estimated_value_inr": 7000000}, extraction) \
        == {"deadline_correct": True, "value_correct": True}
    assert field_accuracy({"deadline": "2026-02-03", "estimated_value_inr": None}, extraction) \
        == {"deadline_correct": False, "value_correct": None}


def test_cost_latency_and_time_saved_and_roi():
    calls = [{"input_tokens": 100, "output_tokens": 50, "latency_ms": 30000, "cost_usd": 0.01},
             {"input_tokens": 10, "output_tokens": 5, "latency_ms": 30000, "cost_usd": 0.0}]
    assert cost_latency(calls) == {"llm_calls": 2, "input_tokens": 110, "output_tokens": 55,
                                   "latency_s": 60.0, "cost_usd": 0.01}
    assert time_saved(120, 1.0, 15) == 104.0
    assert time_saved(None, 1.0, 15) is None             # no measured baseline -> no claim
    assert roi_inr(60, 0.01, 1500, 84.0) == 1499.0
    assert roi_inr(None, 0.01, 1500, 84.0) is None


def test_evaluate_marks_unknowns_as_none():
    label = {"name": "x", "requirements": LABELLED, "deadline": None,
             "estimated_value_inr": None, "expected_recommendation": None, "manual_minutes": None}
    result = {"status": "ok", "extraction": {"requirements": EXTRACTED}, "score":
              {"recommendation": "review"}, "redaction_stats": {"citation_pass_rate": 0.9},
              "calls": []}
    r = evaluate(label, result, {"hourly_rate_inr": 1500, "usd_inr": 84.0, "review_minutes": 15})
    assert r["score_agrees"] is None and r["time_saved_min"] is None and r["roi_inr"] is None
    assert r["citation_pass_rate"] == 0.9 and r["recall"] == 2 / 3


def test_different_certification_with_similar_wording_does_not_match():
    assert match_requirements([{"text": "Bidder must hold a valid CMMI Level 5 certificate"}],
                              [ex("Bidder must hold a valid ISO 27001 certificate")]) == []


def test_paraphrase_with_same_figures_matches():
    lab = [{"text": "Bidder must have 5 years experience in web design"}]
    assert match_requirements(lab, [ex("The bidder must have at least 5 years of experience in "
                                       "web designing, building and maintenance")])


def test_short_label_does_not_match_boilerplate():
    lab = [{"text": "Must have an office in New Delhi"}]
    assert match_requirements(lab, [ex("The agency must hold a valid CMMI Level 5 certificate"),
                                    ex("Average turnover must exceed Rs. 50 lakh")]) == []
    assert match_requirements(lab, [ex("The Bidder must have a registered office/ branch office "
                                       "in New Delhi, India.")])
