"""Streamlit review console (guide §3.2). Talks only to the FastAPI service.

  API_URL=http://127.0.0.1:8000 .venv/bin/streamlit run ui/app.py
"""
import os

import requests
import streamlit as st

API = os.environ.get("API_URL", "http://127.0.0.1:8000")
LONG = 1800  # seconds: a free-tier evaluation can take several minutes

st.set_page_config(page_title="RFP Review Console", layout="wide")


def api(method: str, path: str, **kw):
    try:
        r = requests.request(method, f"{API}{path}", timeout=kw.pop("timeout", 30), **kw)
    except requests.RequestException as e:
        st.error(f"API unreachable at {API}: {type(e).__name__}")
        st.stop()
    return r


def goto(screen: str, run_id: str | None = None):
    """Navigate on the NEXT run: a widget's state (the sidebar radio) cannot be
    changed after it has been drawn in the current run."""
    st.session_state["_goto"] = screen
    if run_id:
        st.session_state.run_id = run_id
    st.rerun()


def badge(grounded):
    return {True: "✅", False: "⚠️"}.get(grounded, "❔")


# ---------------- screens ----------------

def upload_screen():
    st.header("Upload an RFP")
    st.caption("Public or synthetic documents only. Parsing, injection scan and PII redaction run "
               "before any LLM call.")
    f = st.file_uploader("RFP PDF", type=["pdf"])
    if f and st.button("Evaluate", type="primary"):
        with st.spinner("Running pipeline (parse → scan → redact → extract → verify → compliance "
                        "→ score). This can take several minutes on the free tier…"):
            r = api("POST", "/evaluate", files={"file": (f.name, f.getvalue(), "application/pdf")},
                    timeout=LONG)
        if r.status_code != 200:
            st.error(f"{r.status_code}: {r.json().get('detail')}")
        else:
            goto("Run detail", r.json()["run_id"])


def queue_screen():
    st.header("Review queue")
    runs = api("GET", "/runs").json()
    if not runs:
        st.info("No runs awaiting review.")
    for run in runs:
        c1, c2, c3 = st.columns([4, 3, 1])
        c1.write(f"`{run['run_id']}`")
        c2.write(f"{os.path.basename(run['source_file'] or '')} · {run['created_at'][:16]}")
        if c3.button("Open", key=f"open-{run['run_id']}"):
            goto("Run detail", run["run_id"])
    with st.expander("Recently finalized"):
        for run in api("GET", "/runs", params={"status": "final"}).json()[:20]:
            if st.button(f"{run['run_id']} · {run['created_at'][:16]}", key=f"fin-{run['run_id']}"):
                goto("Run detail", run["run_id"])


def review_form(run_id: str, ai_rec: str | None):
    st.subheader("Decision")
    reviewer = st.text_input("Reviewer name", key="reviewer")
    action = st.radio("Action", ["approve", "override"], horizontal=True, key="action")
    body = {"action": action, "reviewer": reviewer.strip()}
    disabled = not reviewer.strip()
    if action == "override":
        recs = ["go", "review", "no_go"]
        body["final_recommendation"] = st.selectbox(
            "Final recommendation", recs, index=recs.index(ai_rec) if ai_rec in recs else 1)
        body["reason"] = st.text_area("Reason (required for an override)", key="reason").strip()
        disabled = disabled or not body["reason"]
        if not body["reason"]:
            st.caption("Submit stays disabled until a written reason is entered.")
    if st.button("Submit decision", type="primary", disabled=disabled, key="submit"):
        r = api("POST", f"/runs/{run_id}/review", json=body)
        if r.status_code == 200:
            st.success(f"Final: {r.json()['review']['final_recommendation']} "
                       f"(notification: {r.json()['notification']})")
            st.rerun()
        else:
            st.error(f"{r.status_code}: {r.json().get('detail')}")


def detail_screen():
    run_id = st.text_input("Run id", value=st.session_state.get("run_id", ""))
    if not run_id:
        st.info("Pick a run from the review queue.")
        return
    r = api("GET", f"/runs/{run_id}")
    if r.status_code != 200:
        st.error(f"{r.status_code}: {r.json().get('detail')}")
        return
    run = r.json()
    # Pending runs: the interrupt payload. Final runs: the same fields from saved state.
    p = run["review_payload"] or {k: run.get(k) for k in (
        "extraction", "compliance", "risk_flags", "errors", "redaction_stats")} | {
        "status": run["status"]}
    final = run["review"] if run["status"] == "final" else None
    ex, score = p.get("extraction") or {}, run.get("score")

    if p.get("status") == "error" or run["status"] == "error":
        st.error("ERROR STATE — not scored. " + " | ".join(p.get("errors") or []))
    for flag in p.get("risk_flags") or []:
        st.warning(f"Prompt-injection heuristic matched: {flag}. Recommendation capped at review.")
    for problem in p.get("previous_decision_problems") or []:
        st.error(f"Previous decision rejected: {problem}")
    if final:
        st.success(f"FINAL: **{final['final_recommendation']}** by {final['reviewer']} "
                   f"({final['action']}) — {final.get('reason') or 'no reason given'}")

    left, right = st.columns([5, 6])
    with left:
        n = st.number_input("Original PDF page", min_value=1, max_value=max(run["pages"], 1),
                            value=st.session_state.get("page", 1))
        img = api("GET", f"/runs/{run_id}/page/{int(n)}")
        if img.status_code == 200:
            st.image(img.content, width="stretch")
    with right:
        if score:
            st.metric("Score", f"{score['total']}", score["recommendation"].upper(),
                      delta_color="off", delta_arrow="off")
            st.bar_chart(score["breakdown"], horizontal=True)
            for reason in score["reasons"]:
                st.caption(f"• {reason}")
        stats = p.get("redaction_stats") or {}
        if stats:
            st.caption("Redactions before LLM: " + ", ".join(
                f"{k} {v}" for k, v in stats.items() if k != "citation_pass_rate")
                + f" · citation pass rate {stats.get('citation_pass_rate', 0):.0%}")
        flags = {f["requirement_id"]: f for f in p.get("compliance") or []}
        reqs = ex.get("requirements", [])
        st.subheader(f"Requirements ({len(reqs)})")
        for req in reqs:
            f = flags.get(req["id"])
            label = (f"{badge(req.get('grounded'))} {req['id']} · {req['category']}"
                     f"{' · mandatory' if req['mandatory'] else ''}"
                     f"{' · ' + f['status'].upper() if f else ''} — {req['text'][:90]}")
            with st.expander(label):
                st.write(req["text"])
                st.markdown(f"> {req['citation']['exact_quote']}")
                if st.button(f"Show page {req['citation']['page_number']}", key=f"pg-{req['id']}"):
                    st.session_state.page = req["citation"]["page_number"]
                    st.rerun()
                if f:
                    st.write(f"**Compliance: {f['status']}** — {f['reasoning']}")
                    st.caption(f"{badge(f.get('evidence_grounded'))} evidence: {f['evidence']}")
                    for ret in f.get("retrieved", []):
                        st.caption(f"retrieved: {ret['section']} (distance {ret['distance']})")
        if run["awaiting_review"]:
            review_form(run_id, (score or {}).get("recommendation"))


def telemetry_screen():
    st.header("Telemetry")
    rows = api("GET", "/telemetry").json()
    if not rows:
        st.info("No LLM calls recorded yet.")
        return
    c = st.columns(4)
    c[0].metric("Runs", len(rows))
    c[1].metric("LLM calls", sum(r["calls"] for r in rows))
    c[2].metric("Tokens in / out", f"{sum(r['input_tokens'] for r in rows):,} / "
                                   f"{sum(r['output_tokens'] for r in rows):,}")
    c[3].metric("Cost (USD)", f"{sum(r['cost_usd'] for r in rows):.4f}")
    st.dataframe(rows, width="stretch")
    st.caption("Time saved needs a measured manual baseline; see the evaluation table in README.")


SCREENS = {"Upload": upload_screen, "Review queue": queue_screen,
           "Run detail": detail_screen, "Telemetry": telemetry_screen}
st.sidebar.title("RFP Intelligence")
st.session_state.setdefault("screen", "Review queue")
if "_goto" in st.session_state:  # apply navigation BEFORE the radio is drawn
    st.session_state.screen = st.session_state.pop("_goto")
choice = st.sidebar.radio("Screen", list(SCREENS), key="screen")
st.sidebar.caption(f"API: {API}")
SCREENS[choice]()
