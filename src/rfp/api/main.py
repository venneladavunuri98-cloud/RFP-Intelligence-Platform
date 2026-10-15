"""FastAPI service (guide §3.1).

Plain `def` handlers: FastAPI runs them in a thread pool, so blocking LLM calls
don't stall the server. Review validation is enforced twice on purpose: here at
the API boundary (422) and again inside the graph (validate_review).

  PYTHONPATH=src .venv/bin/uvicorn rfp.api.main:app --host 127.0.0.1 --port 8000
"""
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Literal, Optional

import pymupdf
import requests
from fastapi import FastAPI, HTTPException, Response, UploadFile
from fastapi.responses import HTMLResponse
from langgraph.types import Command
from pydantic import BaseModel, Field

from .. import config, db
from ..graph import open_graph

log = logging.getLogger("rfp.api")


class ReviewDecision(BaseModel):
    action: Literal["approve", "override"]
    reviewer: str = Field(min_length=1)
    reason: str = ""
    final_recommendation: Optional[Literal["go", "review", "no_go"]] = None


def _cfg(run_id: str) -> dict:
    return {"configurable": {"thread_id": run_id}}


def _pending_payload(state):
    for task in state.tasks:
        if task.interrupts:
            return task.interrupts[0].value
    return None


def notify(event: dict) -> Optional[str]:
    """POST to the n8n webhook. A failed notification never undoes a decision;
    it is logged and reported back to the caller."""
    if not config.N8N_WEBHOOK_URL:
        return "notifications disabled (N8N_WEBHOOK_URL not set)"
    try:
        requests.post(config.N8N_WEBHOOK_URL, json=event, timeout=10).raise_for_status()
        return None
    except requests.RequestException as e:
        log.warning("n8n notification failed: %s", e)
        return f"notification failed: {type(e).__name__}"


def create_app(graph=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if graph is not None:
            app.state.graph = graph
            yield
            return
        if config.STORAGE == "postgres":
            db.init_db()
            if db.capability_count() == 0:  # first start: embed data/capabilities.md (MOCK)
                from ..capabilities import ingest
                try:
                    log.warning("capability store empty: ingested %d sections", ingest())
                except Exception as e:  # never block startup; compliance fails closed meanwhile
                    log.error("capability ingest failed (%s: %s); compliance will score neutral "
                              "until `python -m rfp.capabilities` succeeds", type(e).__name__, e)
        with open_graph() as g:  # Postgres checkpointer on a pool (guide §2.4)
            app.state.graph = g
            yield

    app = FastAPI(title="RFP Intelligence Platform", lifespan=lifespan)

    @app.post("/evaluate")
    def evaluate(file: UploadFile):
        data = file.file.read(config.MAX_UPLOAD_MB * 1024 * 1024 + 1)
        if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(413, f"PDF larger than {config.MAX_UPLOAD_MB} MB")
        if not data.startswith(b"%PDF-"):
            raise HTTPException(415, "Only PDF files are accepted")
        run_id = str(uuid.uuid4())
        config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        path = config.UPLOAD_DIR / f"{run_id}.pdf"  # server-chosen name: no path traversal
        path.write_bytes(data)
        g = app.state.graph
        res = g.invoke({"run_id": run_id, "pdf_path": str(path)}, _cfg(run_id))
        awaiting = "__interrupt__" in res
        if awaiting and config.STORAGE == "postgres":
            db.upsert_run(run_id, "awaiting_review")
        return {"run_id": run_id, "awaiting_review": awaiting, "status": res.get("status"),
                "source_file": file.filename}

    @app.get("/runs")
    def list_runs(status: str = "awaiting_review"):
        if config.STORAGE != "postgres":
            raise HTTPException(501, "Run listing needs Postgres storage")
        return db.runs_with_status(status)

    @app.get("/runs/{run_id}")
    def get_run(run_id: str):
        st = app.state.graph.get_state(_cfg(run_id))
        if not st.values:
            raise HTTPException(404, "Run not found")
        v = st.values
        return {"run_id": run_id, "status": v.get("status"),
                "awaiting_review": _pending_payload(st) is not None,
                "review_payload": _pending_payload(st), "review": v.get("review"),
                "score": v.get("score"), "pages": len(v.get("pages") or {}),
                # also for FINAL runs (no pending interrupt): what the reviewer saw
                "extraction": v.get("extraction"), "compliance": v.get("compliance"),
                "risk_flags": v.get("risk_flags", []), "errors": v.get("errors", []),
                "redaction_stats": v.get("redaction_stats", {}),
                "telemetry": db.run_telemetry(run_id) if config.STORAGE == "postgres" else []}

    @app.post("/runs/{run_id}/review")
    def review(run_id: str, body: ReviewDecision):
        if body.action == "override":
            if not body.reason.strip():
                raise HTTPException(422, "Override requires a written reason")
            if body.final_recommendation is None:
                raise HTTPException(422, "Override requires final_recommendation")
        g = app.state.graph
        st = g.get_state(_cfg(run_id))
        if not st.values:
            raise HTTPException(404, "Run not found")
        if _pending_payload(st) is None:
            raise HTTPException(409, f"Run is not awaiting review (status: {st.values.get('status')})")
        res = g.invoke(Command(resume=body.model_dump()), _cfg(run_id))
        if res.get("status") != "final":  # graph rejected it (second line of defence)
            raise HTTPException(422, {"problems": _pending_payload(g.get_state(_cfg(run_id)))
                                      ["previous_decision_problems"]})
        warning = notify({"event": "review_finalized", "run_id": run_id,
                          "score": res.get("score"), "review": res["review"],
                          "title": ((res.get("extraction") or {}).get("title") or {}).get("value")})
        return {"run_id": run_id, "status": res["status"], "review": res["review"],
                "notification": warning or "sent"}

    @app.get("/runs/{run_id}/page/{n}")
    def page_png(run_id: str, n: int):
        st = app.state.graph.get_state(_cfg(run_id))
        if not st.values:
            raise HTTPException(404, "Run not found")
        try:
            with pymupdf.open(st.values["pdf_path"]) as doc:
                if not 1 <= n <= doc.page_count:
                    raise HTTPException(404, f"Page {n} out of range 1..{doc.page_count}")
                png = doc[n - 1].get_pixmap(dpi=100).tobytes("png")
        except (pymupdf.FileDataError, pymupdf.FileNotFoundError):
            raise HTTPException(404, "Original PDF unavailable")
        return Response(png, media_type="image/png")

    @app.get("/runs/{run_id}/report", response_class=HTMLResponse)
    def report(run_id: str):
        try:
            uuid.UUID(run_id)  # only server-generated ids: no path traversal via run_id
        except ValueError:
            raise HTTPException(404, "Report not found")
        path = config.REPORT_DIR / f"{run_id}.html"
        if not path.is_file():
            raise HTTPException(404, "Report not found (run not final yet?)")
        return HTMLResponse(path.read_text(encoding="utf-8"),
                            headers={"Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'"})

    @app.get("/telemetry")
    def telemetry():
        if config.STORAGE != "postgres":
            raise HTTPException(501, "Telemetry summary needs Postgres storage")
        return db.telemetry_summary()

    @app.get("/health")
    def health():
        return {"ok": True}

    return app


app = create_app()
