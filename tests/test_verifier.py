from rfp.nodes.verifier import is_grounded, verify

PAGES = {
    1: "REQUEST FOR PROPOSAL\nCloud Migration Services for the Department of Revenue",
    2: ("3.1 Eligibility\nThe bidder must hold a valid ISO 27001 certifi-\ncation issued by an "
        "accredited body.\nThe bidder shall   have an average annual turnover of\nRs. 50 crore."),
    3: "The “Successful Bidder” shall submit a Performance Bank Guarantee – 10% of value.",
    4: "Section 2",  # near-empty page
    5: "Queries may be sent to [EMAIL] before the pre-bid meeting.",
}


def cit(page, quote):
    return {"page_number": page, "exact_quote": quote}


def test_exact_quote_passes():
    assert is_grounded(cit(1, "Cloud Migration Services for the Department of Revenue"), PAGES)


def test_hyphenation_and_whitespace_differences_pass():
    q = "The bidder must hold a valid ISO 27001 certification issued by an accredited body."
    assert is_grounded(cit(2, q), PAGES)
    assert is_grounded(cit(2, "shall have an average annual turnover of Rs. 50 crore"), PAGES)


def test_case_differences_pass():
    assert is_grounded(cit(1, "request for proposal cloud migration services"), PAGES)


def test_straightened_quotes_and_dashes_pass():
    q = 'The "Successful Bidder" shall submit a Performance Bank Guarantee - 10% of value.'
    assert is_grounded(cit(3, q), PAGES)


def test_quote_matches_redacted_text_the_llm_saw():
    assert is_grounded(cit(5, "Queries may be sent to [EMAIL] before the pre-bid meeting"), PAGES)


def test_invented_quote_fails():
    assert not is_grounded(cit(2, "The bidder must have CMMI Level 5 and 20 years of experience"),
                           PAGES)


def test_altered_number_fails():
    assert not is_grounded(cit(2, "average annual turnover of Rs. 500 crore"), PAGES)


def test_wrong_page_fails():
    assert not is_grounded(cit(1, "The bidder must hold a valid ISO 27001 certification"), PAGES)


def test_nonexistent_page_fails():
    assert not is_grounded(cit(99, "Cloud Migration Services for the Department"), PAGES)


def test_long_invented_quote_cannot_match_near_empty_page():
    q = "Bidders must comply with Section 2 and offer a 50% discount on all services"
    assert not is_grounded(cit(4, q), PAGES)


def make_state():
    return {
        "pages": PAGES, "redaction_stats": {"EMAIL": 1},
        "extraction": {
            "title": {"value": "Cloud Migration", "grounded": True,  # LLM tried to self-grade
                      "citation": cit(1, "Cloud Migration Services for the Department")},
            "issuer": None,
            "submission_deadline": None,
            "estimated_value_inr": {"value": 5e8, "grounded": None,
                                    "citation": cit(2, "estimated contract value Rs. 50 crore")},
            "requirements": [
                {"id": "R1", "text": "ISO 27001", "category": "eligibility", "mandatory": True,
                 "grounded": None, "citation": cit(2, "must hold a valid ISO 27001 certification")},
                {"id": "R2", "text": "CMMI 5", "category": "eligibility", "mandatory": True,
                 "grounded": True, "citation": cit(2, "must be appraised at CMMI Level 5")},
            ],
        },
    }


def test_verify_sets_grounded_and_pass_rate():
    state = make_state()
    out = verify(state)
    ex = out["extraction"]
    assert ex["title"]["grounded"] is True
    assert ex["estimated_value_inr"]["grounded"] is False      # invented quote
    assert [r["grounded"] for r in ex["requirements"]] == [True, False]  # LLM's True overwritten
    assert out["redaction_stats"] == {"EMAIL": 1, "citation_pass_rate": 0.5}


def test_verify_does_not_mutate_input_state():
    state = make_state()
    verify(state)
    assert state["extraction"]["requirements"][1]["grounded"] is True  # untouched


def test_verify_handles_no_items():
    state = {"pages": PAGES, "extraction": {"title": None, "requirements": []}}
    assert verify(state)["redaction_stats"]["citation_pass_rate"] == 0


def test_truncated_number_fails():
    assert not is_grounded(cit(2, "average annual turnover of Rs. 5"), PAGES)


def test_numbers_split_by_hyphenated_line_break_still_pass():
    pages = {1: "Contract period of 36 months from the date of award of con-\ntract."}
    assert is_grounded(cit(1, "Contract period of 36 months from the date of award of contract"),
                       pages)
