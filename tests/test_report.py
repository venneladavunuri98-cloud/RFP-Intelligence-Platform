from rfp.report import render

STATE = {"run_id": "r1", "pdf_path": "x.pdf",
         "extraction": {"title": {"value": "Cloud <script>alert(1)</script>", "grounded": True,
                                  "citation": {"page_number": 1, "exact_quote": "q" * 10}},
                        "requirements": [{"id": "R1", "text": "Hold ISO 27001", "category": "eligibility",
                                          "mandatory": True, "grounded": True,
                                          "citation": {"page_number": 2, "exact_quote":
                                                       "<img src=x onerror=alert(1)> must hold ISO"}}]},
         "compliance": [{"requirement_id": "R1", "status": "covered", "reasoning": "certified",
                         "evidence": "ISO/IEC 27001:2022 certified", "evidence_grounded": True,
                         "retrieved": [{"section": "Information security certifications"}]}],
         "score": {"total": 75.8, "recommendation": "go", "breakdown": {"technical_fit": 1.0},
                   "reasons": ["2 legal-risk clause(s) flagged"], "rubric_version": 1},
         "redaction_stats": {"EMAIL": 1, "citation_pass_rate": 1.0},
         "risk_flags": ["page 2: matched /system prompt/"]}
REVIEW = {"action": "override", "reviewer": "Vennela", "reason": "Unlimited liability",
          "ai_recommendation": "go", "final_recommendation": "no_go", "decided_at": "2026-10-15"}


def test_report_contains_required_sections():
    html = render(STATE, REVIEW, [{"node": "extract", "attempt": 1, "ok": True, "input_tokens": 10,
                                   "output_tokens": 5, "latency_ms": 900, "cost_usd": 0.0}])
    for needle in ("Final recommendation: no_go", "Reason: Unlimited liability", "Score 75.8",
                   "technical_fit", "must hold ISO", "<td>p2</td>", "covered", "ISO/IEC 27001:2022",
                   "Information security certifications", "matched /system prompt/", "<td>900</td>"):
        assert needle in html, needle


def test_untrusted_rfp_text_is_escaped():
    html = render(STATE, REVIEW, [])
    assert "<script>" not in html and "<img" not in html
    assert "&lt;script&gt;" in html and "&lt;img src=x" in html
