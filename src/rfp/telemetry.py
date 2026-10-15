"""Per-LLM-call telemetry (guide §1.8/§2.4): tokens, latency, cost.

Written to the Postgres audit_llm_calls table (STORAGE="postgres"). Telemetry
must never break a run: if the DB write fails, the record goes to the JSONL
file with the DB error attached, so no measurement is lost."""
import json
import time

from . import config


def log_call(run_id, node, model, raw_msg, started, **extra):
    """`started` is a time.perf_counter() value taken just before the call."""
    u = getattr(raw_msg, "usage_metadata", None) or {}
    inp, out = u.get("input_tokens", 0), u.get("output_tokens", 0)
    price = config.PRICES.get(model, {"in": 0, "out": 0})  # USD per 1M tokens
    cost = (inp * price["in"] + out * price["out"]) / 1e6
    record = {"run_id": run_id, "node": node, "model": model,
              "input_tokens": inp, "output_tokens": out,
              "latency_ms": int((time.perf_counter() - started) * 1000),
              "cost_usd": cost, **extra}
    if config.STORAGE == "postgres":
        from . import db
        try:
            db.insert_llm_call(record)
            return record
        except Exception as e:
            record = {**record, "db_error": f"{type(e).__name__}: {e}"}
    with config.TELEMETRY_LOG.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return record
