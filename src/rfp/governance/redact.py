"""Regex PII redaction (guide §1.4), applied per page before any LLM call.

Limitations: regex does not catch names or addresses (NER is added in
Stage 2). Redaction is not anonymization. A 10-digit number starting 6-9
(e.g. an un-comma'd amount) is indistinguishable from a mobile number and is
redacted as PHONE. The redaction mapping is not stored.
"""
import re
from typing import Callable

# PHONE runs before AADHAAR so "+919876543210" is labelled PHONE, not AADHAAR.
# Lookarounds instead of \b: also reject digits glued to "/" (tender IDs).
RULES = {
    "EMAIL": r"[\w.+-]+@[\w-]+\.[\w.-]+",
    "PAN": r"\b[A-Z]{5}\d{4}[A-Z]\b",
    "PHONE": r"(?<![\w/])(?:\+?91[ -]?)?[6-9]\d{4}[ -]?\d{5}(?![\w/])",
    # Same separator (space, hyphen, or none) between groups, never a newline,
    # so numbers in separate table cells are not joined. UIDAI numbers never start 0/1.
    "AADHAAR": r"(?<![\w/])[2-9]\d{3}([ -]?)\d{4}\1\d{4}(?![\w/])",
}

# Verhoeff check-digit tables (the checksum used by Aadhaar numbers)
_D = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
      [2, 3, 4, 0, 1, 7, 8, 9, 5, 6], [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
      [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
      [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
      [8, 7, 6, 5, 9, 3, 2, 1, 0, 4], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
_P = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
      [5, 8, 0, 3, 7, 9, 6, 1, 4, 2], [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
      [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
      [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]]


def verhoeff_valid(digits: str) -> bool:
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _D[c][_P[i % 8][int(ch)]]
    return c == 0


def _is_aadhaar(match: str) -> bool:
    digits = re.sub(r"\D", "", match)
    groups = [digits[i:i + 4] for i in (0, 4, 8)]
    if all(g[:2] in ("19", "20") for g in groups):  # e.g. a "2023 2024 2025" table row
        return False
    return verhoeff_valid(digits)


VALIDATORS: dict[str, Callable[[str], bool]] = {"AADHAAR": _is_aadhaar}


def redact(text: str) -> tuple[str, dict]:
    stats = {}
    for label, pat in RULES.items():
        valid = VALIDATORS.get(label, lambda _: True)
        count = 0

        def repl(m: re.Match) -> str:
            nonlocal count
            if not valid(m.group()):
                return m.group()
            count += 1
            return f"[{label}]"

        text = re.sub(pat, repl, text)
        stats[label] = count
    return text, stats
