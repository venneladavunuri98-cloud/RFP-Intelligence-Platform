"""Prompt-injection scan (guide §1.4).

HEURISTIC ONLY: a regex screen for common injection phrasings. It is not a
security guarantee; determined adversarial text can bypass it. Findings are
routed to risk_flags, shown to the reviewer, and cap the recommendation at
"review". Defence in depth comes from <rfp> delimiters, a tool-less
extractor, structured output, and mandatory human approval.
"""
import re
import unicodedata

PATTERNS = [
    r"ignore (all )?(previous|prior|above) instructions",
    r"disregard (the )?(system|previous) (prompt|instructions)",
    r"you are now\b", r"system prompt", r"reveal (your|the) (prompt|instructions)",
    r"do not (tell|inform) the (user|reviewer)",
]

_INVISIBLE = dict.fromkeys(map(ord, "\u00ad\u200b\u200c\u200d\u2060\ufeff"))


def normalize(text: str) -> str:
    """Undo PDF line wrapping and simple obfuscation so phrases split across
    lines, padded with extra spaces, or broken by zero-width chars still match."""
    text = unicodedata.normalize("NFKC", text).translate(_INVISIBLE)
    return re.sub(r"\s+", " ", text)


def scan(pages: dict[int, str]) -> list[str]:
    findings = []
    for n, raw in pages.items():
        text = normalize(raw)
        for p in PATTERNS:
            if re.search(p, text, re.I):
                findings.append(f"page {n}: matched /{p}/")
    return findings
