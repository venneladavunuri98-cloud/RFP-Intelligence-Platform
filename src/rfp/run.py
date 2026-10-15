"""Terminal runner (guide §1.9/§2.4).

  python -m rfp.run <pdf> [--thread-id ID]   start a run
  python -m rfp.run --thread-id ID           resume a run waiting for review

Runs the graph until the human-review interrupt, prints the draft, asks the
reviewer for a decision, and resumes until the run is final. With Postgres
checkpoints the process can die at the review prompt and be resumed later.
"""
import argparse
import json
import uuid

from langgraph.types import Command

from . import config
from .graph import RECOMMENDATIONS, open_graph

BADGE = {True: "OK ", False: "!! ", None: "?  "}


def render_draft(p: dict) -> str:
    out = [f"\n=== DRAFT FOR REVIEW (status: {p['status']}) ==="]
    for problem in p.get("previous_decision_problems") or []:
        out.append(f"  REJECTED DECISION: {problem}")
    for e in p.get("errors") or []:
        out.append(f"  ERROR: {e}")
    for f in p.get("risk_flags") or []:
        out.append(f"  RISK FLAG: {f}")
    out.append(f"  Redaction/citations: {json.dumps(p.get('redaction_stats') or {})}")
    ex = p.get("extraction")
    if ex:
        for k in ("title", "issuer", "submission_deadline", "estimated_value_inr"):
            f = ex.get(k)
            if f:
                out.append(f"  {BADGE[f.get('grounded')]}{k}: {f['value']} "
                           f"(p{f['citation']['page_number']})")
        out.append("  Requirements (OK = grounded, !! = quote not found on page):")
        for r in ex["requirements"]:
            out.append(f"    {BADGE[r.get('grounded')]}{r['id']} [{r['category']}"
                       f"{', mandatory' if r['mandatory'] else ''}] p{r['citation']['page_number']}: "
                       f"{r['text']}")
            out.append(f"        quote: \"{r['citation']['exact_quote']}\"")
    flags = p.get("compliance") or []
    if flags:
        out.append("  Compliance (vs capabilities.md; evidence OK = found in retrieved passage):")
        for f in flags:
            out.append(f"    {f['requirement_id']}: {f['status'].upper()} - {f['reasoning']}")
            out.append(f"        {BADGE[f.get('evidence_grounded')]}evidence: \"{f['evidence']}\"")
            out.append("        retrieved: " + "; ".join(
                f"{r['section']} ({r['distance']})" for r in f.get("retrieved", [])))
    s = p.get("score")
    if s:
        out.append(f"  SCORE {s['total']} -> {s['recommendation'].upper()} "
                   f"(rubric v{s['rubric_version']})")
        out.append(f"  Breakdown: {json.dumps(s['breakdown'])}")
        for reason in s["reasons"]:
            out.append(f"    - {reason}")
    return "\n".join(out)


def ask_decision(ask=input) -> dict:
    reviewer = ask("Reviewer name: ").strip()
    action = ask("Action [approve/override]: ").strip().lower()
    decision = {"action": action, "reviewer": reviewer, "reason": ""}
    if action == "override":
        decision["final_recommendation"] = ask(f"Final recommendation {RECOMMENDATIONS}: ").strip()
        decision["reason"] = ask("Reason (required): ").strip()
    return decision


def _pending(graph, cfg):
    """Interrupt payload of a run waiting for review, else None."""
    st = graph.get_state(cfg)
    for task in st.tasks:
        if task.interrupts:
            return task.interrupts[0].value
    return None


def _mark(run_id: str, status: str) -> None:
    if config.STORAGE == "postgres":
        from . import db
        db.upsert_run(run_id, status)


def run(pdf_path: str | None, thread_id: str, ask=input, echo=print) -> dict:
    cfg = {"configurable": {"thread_id": thread_id}}
    if config.STORAGE == "postgres":
        from . import db
        db.init_db()  # idempotent: applies any missing tables/columns before writing
    with open_graph() as graph:
        st = graph.get_state(cfg)
        if st.values and not st.next:                       # already finished
            echo(f"Run {thread_id} is already {st.values.get('status')}.")
            return st.values
        if st.values:                                        # resume after restart
            echo(f"Resuming run {thread_id} from its checkpoint.")
        else:
            if not pdf_path:
                raise SystemExit(f"No run {thread_id!r} found; pass a PDF to start one.")
            graph.invoke({"run_id": thread_id, "pdf_path": pdf_path}, cfg)
        while (payload := _pending(graph, cfg)) is not None:
            _mark(thread_id, "awaiting_review")
            echo(render_draft(payload))
            graph.invoke(Command(resume=ask_decision(ask)), cfg)
        res = graph.get_state(cfg).values
    echo(f"\n=== FINAL: {res['review']['final_recommendation'].upper()} ===")
    echo(json.dumps(res["review"], indent=2))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate one RFP PDF with human review.")
    ap.add_argument("pdf", nargs="?", help="PDF to evaluate (omit to resume --thread-id)")
    ap.add_argument("--thread-id", default=None)
    args = ap.parse_args(argv)
    if not args.pdf and not args.thread_id:
        ap.error("pass a PDF to start a run, or --thread-id to resume one")
    run(args.pdf, args.thread_id or f"run-{uuid.uuid4().hex[:8]}")


if __name__ == "__main__":
    main()
