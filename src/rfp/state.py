"""LangGraph state (guide §1.2).

Store plain dicts (model.model_dump()), not model objects or PDF handles,
so checkpoints serialize cleanly.
"""
from typing import TypedDict


class GraphState(TypedDict, total=False):
    run_id: str
    pdf_path: str
    pages: dict[int, str]  # redacted text per page
    redaction_stats: dict
    risk_flags: list[str]  # injection findings etc.
    extraction: dict
    compliance: list[dict]
    score: dict
    review: dict  # human decision
    status: str  # "ok" | "error" | "final"
    errors: list[str]
