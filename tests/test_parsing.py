import pytest

from tests.conftest import make_pdf
from rfp.parsing import PDFParseError, ScannedPDFError, parse_pdf

FILLER = "The bidder shall provide cloud migration services as described herein. "


def test_pages_are_one_based_and_keep_their_text(tmp_path):
    pdf = make_pdf(tmp_path / "rfp.pdf", [f"PAGE-ONE {FILLER}", f"PAGE-TWO {FILLER}",
                                         f"PAGE-THREE {FILLER}"])
    pages = parse_pdf(pdf)
    assert list(pages) == [1, 2, 3]
    assert "PAGE-ONE" in pages[1] and "PAGE-TWO" in pages[2] and "PAGE-THREE" in pages[3]
    assert "PAGE-TWO" not in pages[1]


def test_blank_page_inside_text_pdf_is_kept(tmp_path):
    pdf = make_pdf(tmp_path / "rfp.pdf", [FILLER * 3, "", FILLER * 3])
    pages = parse_pdf(pdf)
    assert pages[2].strip() == "" and len(pages) == 3  # numbering not shifted


def test_image_only_pdf_raises_scanned_error(tmp_path):
    pdf = make_pdf(tmp_path / "scan.pdf", ["", ""])
    with pytest.raises(ScannedPDFError):
        parse_pdf(pdf)


def test_corrupt_file_raises_parse_error(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"this is not a pdf")
    with pytest.raises(PDFParseError):
        parse_pdf(str(bad))


def test_missing_file_raises_parse_error(tmp_path):
    with pytest.raises(PDFParseError):
        parse_pdf(str(tmp_path / "missing.pdf"))


def test_scanned_error_is_a_parse_error():
    # ingest catches PDFParseError, so scanned PDFs still land in the error state
    assert issubclass(ScannedPDFError, PDFParseError)
