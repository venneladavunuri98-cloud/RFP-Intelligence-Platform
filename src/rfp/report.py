"""Final HTML report (guide §3.4): summary, score breakdown, every requirement with
quote and page, compliance evidence, reviewer decision and reason, run telemetry.

All RFP-derived text is untrusted: every value is HTML-escaped (no raw HTML, no scripts).
"""
from html import escape


def _e(v) -> str:
    return escape("" if v is None else str(v))


def render(state: dict, review: dict, telemetry: list[dict]) -> str:
    ex = state.get("extraction") or {}
    score = state.get("score") or {}
    flags = {f["requirement_id"]: f for f in state.get("compliance") or []}
    stats = state.get("redaction_stats") or {}

    def field(key):
        f = ex.get(key)
        if not f:
            return "<td>—</td><td></td>"
        mark = "✅" if f.get("grounded") else "⚠️ ungrounded"
        return f"<td>{_e(f['value'])}</td><td>p{_e(f['citation']['page_number'])} {mark}</td>"

    rows = []
    for r in ex.get("requirements", []):
        f = flags.get(r["id"])
        comp = (f"<b>{_e(f['status'])}</b>: {_e(f['reasoning'])}<br><small>evidence "
                f"{'✅' if f.get('evidence_grounded') else '⚠️ unverified'}: {_e(f['evidence'])}"
                f"<br>retrieved: {_e('; '.join(x['section'] for x in f.get('retrieved', [])))}</small>"
                if f else "—")
        rows.append(f"<tr><td>{_e(r['id'])}</td><td>{_e(r['category'])}"
                    f"{' (mandatory)' if r['mandatory'] else ''}</td><td>{_e(r['text'])}"
                    f"<blockquote>{_e(r['citation']['exact_quote'])}</blockquote></td>"
                    f"<td>p{_e(r['citation']['page_number'])}</td>"
                    f"<td>{'✅' if r.get('grounded') else '⚠️ excluded'}</td><td>{comp}</td></tr>")
    breakdown = "".join(f"<tr><td>{_e(k)}</td><td>{_e(v)}</td></tr>"
                        for k, v in (score.get("breakdown") or {}).items())
    reasons = "".join(f"<li>{_e(x)}</li>" for x in score.get("reasons", []))
    flags_html = "".join(f"<li>{_e(x)}</li>" for x in state.get("risk_flags") or [])
    errors_html = "".join(f"<li>{_e(x)}</li>" for x in state.get("errors") or [])
    tel = "".join(f"<tr><td>{_e(c['node'])}</td><td>{_e(c.get('attempt'))}</td><td>{_e(c.get('ok'))}</td>"
                  f"<td>{_e(c['input_tokens'])}</td><td>{_e(c['output_tokens'])}</td>"
                  f"<td>{_e(c['latency_ms'])}</td><td>{_e(c['cost_usd'])}</td></tr>" for c in telemetry)
    title = (ex.get("title") or {}).get("value") or state.get("run_id")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RFP report: {_e(title)}</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#1a1a1a}}
table{{border-collapse:collapse;width:100%;margin:.5rem 0 1.5rem}}td,th{{border:1px solid #ccc;padding:.4rem;vertical-align:top;text-align:left}}
blockquote{{margin:.3rem 0;padding-left:.6rem;border-left:3px solid #999;color:#444}}.warn{{background:#fff4e5;padding:.6rem}}</style>
</head><body>
<h1>{_e(title)}</h1>
<p>Run <code>{_e(state.get('run_id'))}</code> · source <code>{_e(state.get('pdf_path'))}</code></p>
<h2>Decision</h2>
<p><b>Final recommendation: {_e(review.get('final_recommendation'))}</b>
 (AI: {_e(review.get('ai_recommendation'))}) · {_e(review.get('action'))} by {_e(review.get('reviewer'))}
 at {_e(review.get('decided_at'))}<br>Reason: {_e(review.get('reason') or '—')}</p>
{f'<div class="warn"><b>Errors</b><ul>{errors_html}</ul></div>' if errors_html else ''}
{f'<div class="warn"><b>Prompt-injection heuristic flags</b><ul>{flags_html}</ul></div>' if flags_html else ''}
<h2>Summary</h2><table>
<tr><th>Issuer</th>{field('issuer')}</tr><tr><th>Submission deadline</th>{field('submission_deadline')}</tr>
<tr><th>Estimated value (INR)</th>{field('estimated_value_inr')}</tr></table>
<h2>Score {_e(score.get('total', 'n/a'))} → {_e(score.get('recommendation', 'n/a'))} (rubric v{_e(score.get('rubric_version', '-'))})</h2>
<table><tr><th>Dimension</th><th>Score (0–1)</th></tr>{breakdown}</table><ul>{reasons}</ul>
<p>Redactions before LLM: {_e({k: v for k, v in stats.items() if k != 'citation_pass_rate'})} ·
citation pass rate {_e(round(stats.get('citation_pass_rate', 0) * 100))}%</p>
<h2>Requirements ({len(rows)})</h2>
<table><tr><th>ID</th><th>Category</th><th>Requirement and verbatim quote</th><th>Page</th><th>Grounded</th><th>Compliance</th></tr>{''.join(rows)}</table>
<h2>Telemetry</h2>
<table><tr><th>Node</th><th>Attempt</th><th>OK</th><th>Tokens in</th><th>Tokens out</th><th>Latency ms</th><th>Cost USD</th></tr>{tel}</table>
<p><small>Citations prove a quote exists on the cited page, not that it supports the claim. Compliance is
judged against a MOCK capabilities statement. Scores are deterministic (rubric.yaml).</small></p>
</body></html>"""
