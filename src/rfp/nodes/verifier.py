"""Programmatic citation verification (guide §1.6): Python, not an LLM.

Checks each quote against the REDACTED page text — the same text the LLM saw.
Limitation: this proves the quote exists on that page, not that it supports
the extracted claim.
"""
import copy
import re
import unicodedata

from rapidfuzz import fuzz

FUZZY_THRESHOLD = 95  # guide: fix mismatches in norm(), don't go below ~90

# Typographic characters that NFKC leaves alone but LLMs routinely "straighten"
_TYPOGRAPHY = str.maketrans({"‘": "'", "’": "'", "‚": "'", "‛": "'",
                             "“": '"', "”": '"', "„": '"', "‟": '"',
                             "‐": "-", "‑": "-", "‒": "-", "–": "-",
                             "—": "-", "−": "-"})


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).replace("­", "").translate(_TYPOGRAPHY)
    s = re.sub(r"-\s*\n\s*", "", s)  # de-hyphenate line breaks
    return re.sub(r"\s+", " ", s).casefold().strip()


def _numbers(s: str) -> list[str]:
    return re.findall(r"\d+", s)


def is_grounded(citation: dict, pages: dict[int, str]) -> bool:
    page = pages.get(citation["page_number"])
    if not page:
        return False
    q, p = norm(citation["exact_quote"]), norm(page)
    # Fuzzy tolerance is for whitespace/typography, never for figures:
    # "Rs. 500 crore" must not match "Rs. 50 crore" (one char, ~97% similar).
    page_numbers = set(_numbers(p))
    if any(n not in page_numbers for n in _numbers(q)):
        return False
    if q in p:
        return True
    # partial_ratio aligns the SHORTER string inside the longer one; a quote
    # longer than its page would let a near-empty page "match" any invention.
    return len(q) <= len(p) and fuzz.partial_ratio(q, p) >= FUZZY_THRESHOLD


def verify(state):
    ex, pages = copy.deepcopy(state["extraction"]), state["pages"]
    items = list(ex["requirements"])
    items += [ex[k] for k in ("title", "issuer", "submission_deadline", "estimated_value_inr")
              if ex.get(k)]
    for it in items:  # overwrites any value the LLM may have put in `grounded`
        it["grounded"] = is_grounded(it["citation"], pages)
    total = len(items)
    ok = sum(i["grounded"] for i in items)
    return {"extraction": ex, "redaction_stats": {**state.get("redaction_stats", {}),
            "citation_pass_rate": ok / total if total else 0}}
