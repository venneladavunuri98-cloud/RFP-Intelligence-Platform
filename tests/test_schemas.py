import pytest
from pydantic import ValidationError

from rfp.schemas import Citation, ExtractedRequirement, RFPExtraction, RFPScore

CIT = {"page_number": 2, "exact_quote": "Vendor must hold ISO 27001 certification"}


def minimal_extraction(**overrides):
    data = {
        "title": {"value": "Cloud Migration Services", "citation": CIT},
        "requirements": [{"id": "R1", "text": "Hold ISO 27001", "category": "eligibility",
                          "mandatory": True, "citation": CIT}],
    }
    data.update(overrides)
    return data


def test_valid_extraction_and_optional_defaults():
    ex = RFPExtraction.model_validate(minimal_extraction())
    assert ex.issuer is None and ex.submission_deadline is None and ex.estimated_value_inr is None
    assert ex.requirements[0].grounded is None  # only the verifier sets this


def test_dump_roundtrip_is_plain_dict():
    ex = RFPExtraction.model_validate(minimal_extraction())
    dumped = ex.model_dump()
    assert isinstance(dumped["requirements"][0]["citation"], dict)
    assert RFPExtraction.model_validate(dumped) == ex


@pytest.mark.parametrize("bad", [
    {"page_number": 0, "exact_quote": "long enough quote"},   # pages are 1-based
    {"page_number": 1, "exact_quote": "too short"},           # < 10 chars
])
def test_citation_rejects_invalid(bad):
    with pytest.raises(ValidationError):
        Citation.model_validate(bad)


def test_requirement_rejects_unknown_category():
    with pytest.raises(ValidationError):
        ExtractedRequirement.model_validate({"id": "R1", "text": "x", "category": "pricing",
                                             "mandatory": True, "citation": CIT})


def test_extraction_requires_title_and_requirements():
    with pytest.raises(ValidationError):
        RFPExtraction.model_validate({"requirements": []})
    with pytest.raises(ValidationError):
        RFPExtraction.model_validate({"title": {"value": "T", "citation": CIT}})


def test_score_rejects_unknown_recommendation():
    with pytest.raises(ValidationError):
        RFPScore(total=80, breakdown={}, recommendation="maybe", reasons=[], rubric_version=1)
