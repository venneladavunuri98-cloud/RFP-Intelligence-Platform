"""LangGraph pipeline with human review (guide §1.9). Stage 1 uses MemorySaver.

ingest -> extract -> verify -> compliance -> score -> human_review -> finalize
Errors from ingest/extract skip straight to human_review (fail closed); an
invalid review decision loops back to human_review, which interrupts again.
"""
from contextlib import contextmanager
from datetime import datetime, timezone

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from . import config
from .nodes.scorer import score_node
from .state import GraphState

ACTIONS = ("approve", "override")
RECOMMENDATIONS = ("go", "review", "no_go")


def ingest(state):
    from .governance import injection, redact
    from .parsing import PDFParseError, parse_pdf
    if config.STORAGE == "postgres":  # audit: every run is registered before any work
        from . import db
        db.upsert_run(state["run_id"], "processing", source_file=state["pdf_path"])
    try:
        raw = parse_pdf(state["pdf_path"])
    except PDFParseError as e:  # scanned, corrupt or missing -> error state, never scored
        return {"status": "error", "errors": [str(e)], "pages": {}}
    flags = injection.scan(raw)  # scan raw text
    pages, stats = {}, {}
    for n, t in raw.items():  # redact per page so page numbers stay intact
        pages[n], s = redact.redact(t)
        for k, v in s.items():
            stats[k] = stats.get(k, 0) + v
    return {"pages": pages, "risk_flags": flags, "redaction_stats": stats, "status": "ok"}


def review_problems(review) -> list[str]:
    """Pure validation of a reviewer decision; empty list means valid."""
    if not isinstance(review, dict):
        return ["decision must be an object with action and reviewer"]
    problems = []
    if review.get("action") not in ACTIONS:
        problems.append(f"action must be one of {ACTIONS}")
    if not str(review.get("reviewer") or "").strip():
        problems.append("reviewer name is required")
    if review.get("action") == "override":
        if not str(review.get("reason") or "").strip():
            problems.append("override requires a written reason")
        if review.get("final_recommendation") not in RECOMMENDATIONS:
            problems.append(f"override requires final_recommendation in {RECOMMENDATIONS}")
    return problems


def human_review(state):  # NO side effects before interrupt: on resume this node re-runs
    previous = state.get("review")
    decision = interrupt({"status": state.get("status"), "errors": state.get("errors", []),
                          "risk_flags": state.get("risk_flags", []),
                          "redaction_stats": state.get("redaction_stats", {}),
                          "extraction": state.get("extraction"),
                          "compliance": state.get("compliance"), "score": state.get("score"),
                          "previous_decision_problems":
                              review_problems(previous) if previous is not None else []})
    return {"review": decision}


def validate_review(state):
    return "finalize" if not review_problems(state["review"]) else "human_review"


def finalize(state):
    # Audit row written here (never in human_review); Stage 3: generate report.
    # A DB failure raises: the run stays resumable at finalize, never "final" unaudited.
    r = state["review"]
    if state.get("status") == "error":
        final = "manual_review_required"
    elif r["action"] == "override":
        final = r["final_recommendation"]
    else:
        final = state["score"]["recommendation"]
    review = {**r, "ai_recommendation": (state.get("score") or {}).get("recommendation"),
              "final_recommendation": final, "decided_at": datetime.now(timezone.utc).isoformat()}
    telemetry = []
    if config.STORAGE == "postgres":
        from . import db
        db.write_review(state["run_id"], review, state.get("score"))
        db.upsert_run(state["run_id"], "final")
        telemetry = db.run_telemetry(state["run_id"])
    from . import report
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORT_DIR / f"{state['run_id']}.html").write_text(
        report.render(state, review, telemetry), encoding="utf-8")
    return {"status": "final", "review": review}


def after_ingest(s):
    return "human_review" if s["status"] == "error" else "extract"


def after_extract(s):
    return "human_review" if s["status"] == "error" else "verify"


def build(checkpointer=None):
    from .nodes import compliance, extractor, verifier
    g = StateGraph(GraphState)
    for name, fn in [("ingest", ingest), ("extract", extractor.extract), ("verify", verifier.verify),
                     ("compliance", compliance.check), ("score", score_node),
                     ("human_review", human_review), ("finalize", finalize)]:
        g.add_node(name, fn)
    g.add_edge(START, "ingest")
    g.add_conditional_edges("ingest", after_ingest, ["extract", "human_review"])
    g.add_conditional_edges("extract", after_extract, ["verify", "human_review"])
    g.add_edge("verify", "compliance")
    g.add_edge("compliance", "score")
    g.add_edge("score", "human_review")
    g.add_conditional_edges("human_review", validate_review, ["finalize", "human_review"])
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer or MemorySaver())


@contextmanager
def open_graph(storage: str | None = None):
    """Graph with the configured checkpointer: Postgres (durable) or MemorySaver."""
    if (storage or config.STORAGE) == "postgres":
        from . import db
        with db.checkpointer() as cp:
            yield build(cp)
    else:
        yield build(MemorySaver())
