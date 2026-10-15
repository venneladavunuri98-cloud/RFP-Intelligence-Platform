"""Evaluation runner (guide §2.5).

  PYTHONPATH=src:. python -m eval.run_eval              # use cached results where present
  PYTHONPATH=src:. python -m eval.run_eval --refresh mea_pbd   # re-run the pipeline for one RFP

Each labelled RFP is run through the graph up to the human-review interrupt (no
approval is faked); the draft, score and telemetry are cached in eval/results/<name>.json
so metrics can be recomputed without spending LLM quota. Eval runs are marked
runs.status='eval' so they never appear in the human review queue.
"""
import argparse
import json
import pathlib
import uuid

from rfp import config
from eval.metrics import evaluate

LABELS = config.ROOT / "data" / "eval" / "labels"
RESULTS = config.ROOT / "eval" / "results"
ASSUMPTIONS = {  # ASSUMPTIONS, not facts: change them and the ROI column changes
    "hourly_rate_inr": 1500, "usd_inr": 84.0, "review_minutes": 15,
}


def run_pipeline(label: dict) -> dict:
    from rfp import db
    from rfp.graph import open_graph
    db.init_db()
    with db.connect() as conn:  # an eval run interrupted mid-pipeline resumes from its checkpoint
        row = conn.execute("SELECT run_id FROM runs WHERE run_id LIKE %s AND status = 'processing' "
                           "ORDER BY created_at DESC LIMIT 1", (f"eval-{label['name']}-%",)).fetchone()
    run_id = row[0] if row else f"eval-{label['name']}-{uuid.uuid4().hex[:6]}"
    cfg = {"configurable": {"thread_id": run_id}}
    with open_graph() as graph:
        if row and graph.get_state(cfg).next:
            print(f"  resuming {run_id} from checkpoint at {graph.get_state(cfg).next}", flush=True)
            graph.invoke(None, cfg)
        else:
            graph.invoke({"run_id": run_id, "pdf_path": str(config.ROOT / label["pdf"])}, cfg)
        values = graph.get_state(cfg).values
    db.upsert_run(run_id, "eval")
    with db.connect() as conn:
        rows = conn.execute("SELECT node, attempt, ok, input_tokens, output_tokens, latency_ms, "
                            "cost_usd FROM audit_llm_calls WHERE run_id = %s ORDER BY id",
                            (run_id,)).fetchall()
    calls = [dict(zip(("node", "attempt", "ok", "input_tokens", "output_tokens", "latency_ms",
                       "cost_usd"), (*r[:6], float(r[6] or 0)))) for r in rows]
    return {"run_id": run_id, "model": config.MODEL, "status": values.get("status"),
            "errors": values.get("errors", []), "extraction": values.get("extraction"),
            "redaction_stats": values.get("redaction_stats"), "score": values.get("score"),
            "compliance": values.get("compliance"), "calls": calls}


def fmt(v, pct=False):
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    return f"{v:.0%}" if pct else str(v)


def table(rows: list[dict]) -> str:
    head = ["RFP", "Status", "Req. recall", "Req. precision", "Labelled/Extracted", "Deadline",
            "Value", "Citation pass", "Rec. (expected)", "LLM calls", "Tokens in/out",
            "Latency s", "Cost $", "Time saved min", "ROI ₹ (assumed)"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        lines.append("| " + " | ".join([
            r["name"], r["status"] or "n/a", fmt(r["recall"], True), fmt(r["precision"], True),
            f"{r['labelled']}/{r['extracted']}", fmt(r["deadline_correct"]), fmt(r["value_correct"]),
            fmt(r["citation_pass_rate"], True),
            f"{fmt(r['recommendation'])} ({fmt(r['expected_recommendation'])})",
            str(r["llm_calls"]), f"{r['input_tokens']}/{r['output_tokens']}", str(r["latency_s"]),
            str(r["cost_usd"]), fmt(r["time_saved_min"]), fmt(r["roi_inr"])]) + " |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", nargs="*", default=None,
                    help="re-run the pipeline for these RFP names (no names = all)")
    args = ap.parse_args(argv)
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(LABELS.glob("*.json")):
        label = json.loads(path.read_text())
        cache = RESULTS / f"{label['name']}.json"
        refresh = args.refresh is not None and (not args.refresh or label["name"] in args.refresh)
        if refresh or not cache.exists():
            print(f"running pipeline for {label['name']} ...", flush=True)
            cache.write_text(json.dumps(run_pipeline(label), indent=1, default=str))
        rows.append(evaluate(label, json.loads(cache.read_text()), ASSUMPTIONS))
    out = table(rows)
    (RESULTS / "summary.md").write_text(out + "\n\nAssumptions: " + json.dumps(ASSUMPTIONS) + "\n")
    print(out)
    return rows


if __name__ == "__main__":
    main()
