import pytest

from rfp.governance.redact import redact, verhoeff_valid

VALID_AADHAAR = "2345 6789 0124"  # synthetic number with a valid Verhoeff check digit


@pytest.mark.parametrize("text,label", [
    ("Contact procurement.cell@nic.in for queries", "EMAIL"),
    ("PAN: ABCDE1234F", "PAN"),
    ("Call 9876543210 today", "PHONE"),
    ("Call +91 9876543210", "PHONE"),
    ("Call +919876543210", "PHONE"),
    ("Call 98765 43210", "PHONE"),
    ("Call +91-98765-43210", "PHONE"),
    (f"Aadhaar {VALID_AADHAAR}", "AADHAAR"),
    (f"Aadhaar {VALID_AADHAAR.replace(' ', '-')}", "AADHAAR"),
    (f"Aadhaar {VALID_AADHAAR.replace(' ', '')}", "AADHAAR"),
])
def test_pii_is_masked(text, label):
    out, stats = redact(text)
    assert f"[{label}]" in out
    assert stats[label] == 1
    assert sum(stats.values()) == 1  # labelled once, not double-counted


@pytest.mark.parametrize("text", [
    "Estimated value Rs. 2,50,00,000 (INR 25000000)",
    "Last date of submission: 15/11/2026 at 15:00 hrs",
    "Year\n2023\n2024\n2025\nTurnover",          # table column: never join across lines
    "Year 2023 2024 2025",                        # table row: year-like groups
    "FY 2023-24\n1250\n3400\n5600 lakhs",
    "Tender No. GEM/2026/B/123456789012",         # IDs glued to '/'
    "Aadhaar 2345 6789 0120",                     # bad checksum: not a real Aadhaar
    "EMD of Rs. 5,00,000 valid for 180 days",
])
def test_normal_numbers_are_not_masked(text):
    out, stats = redact(text)
    assert out == text
    assert sum(stats.values()) == 0


def test_stats_report_every_rule_and_count_multiple_hits():
    out, stats = redact("a@b.com and c@d.org, call 9876543210")
    assert stats == {"EMAIL": 2, "PAN": 0, "PHONE": 1, "AADHAAR": 0}
    assert out == "[EMAIL] and [EMAIL], call [PHONE]"


def test_redaction_keeps_line_structure():
    text = "Line one\nContact: x@y.com\nLine three"
    assert redact(text)[0].count("\n") == 2


def test_verhoeff():
    assert verhoeff_valid("234567890124")
    assert not verhoeff_valid("234567890120")


def test_known_limitation_ten_digit_amount_looks_like_phone():
    # Documented limitation: an un-comma'd 10-digit amount starting 6-9 is redacted.
    assert redact("Rs 7500000000 total")[0] == "Rs [PHONE] total"
