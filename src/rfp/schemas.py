"""Typed data contracts for all agent input/output (guide §1.1).

The LLM fills RFPExtraction; Python fills every `grounded` field (verifier),
ComplianceResult (Stage 2) and RFPScore (scorer).
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field


class Citation(BaseModel):
    page_number: int = Field(ge=1)
    exact_quote: str = Field(min_length=10)


class TextField(BaseModel):
    value: str
    citation: Citation
    grounded: Optional[bool] = None


class NumericField(BaseModel):
    value: float
    citation: Citation
    grounded: Optional[bool] = None


class ExtractedRequirement(BaseModel):
    id: str  # "R1", "R2", ...
    text: str
    category: Literal["technical", "eligibility", "commercial",
                      "timeline", "legal_risk", "other"]
    mandatory: bool
    citation: Citation
    grounded: Optional[bool] = None  # set by verifier, never by LLM


class RFPExtraction(BaseModel):
    title: TextField
    issuer: Optional[TextField] = None
    submission_deadline: Optional[TextField] = None  # ISO date YYYY-MM-DD
    estimated_value_inr: Optional[NumericField] = None
    requirements: list[ExtractedRequirement]


class ComplianceFlag(BaseModel):
    requirement_id: str
    status: Literal["covered", "partial", "gap"]
    evidence: str  # copied from retrieved capability passage
    reasoning: str


class ComplianceResult(BaseModel):
    flags: list[ComplianceFlag]


class RFPScore(BaseModel):
    total: float
    breakdown: dict[str, float]
    recommendation: Literal["go", "review", "no_go"]
    reasons: list[str]
    rubric_version: int
