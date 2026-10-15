"""Page-level PDF parsing (guide §1.3).

Returns {page_number: text} with 1-based page numbers that are preserved
through redaction, extraction, and citation verification.
"""
import pymupdf  # PyMuPDF; `import fitz` is deprecated since 1.28

MIN_AVG_CHARS_PER_PAGE = 50  # tune on your samples


class PDFParseError(Exception):
    """Any PDF the pipeline cannot read; ingest routes it to the error state."""


class ScannedPDFError(PDFParseError):
    """Little/no text layer: likely a scanned (image-only) PDF. No OCR in this version."""


def parse_pdf(path: str) -> dict[int, str]:
    try:
        with pymupdf.open(path) as doc:
            pages = {i + 1: p.get_text("text") for i, p in enumerate(doc)}
    except (pymupdf.FileDataError, pymupdf.FileNotFoundError) as e:
        raise PDFParseError(f"Cannot open PDF: {e}") from e
    avg = sum(len(t.strip()) for t in pages.values()) / max(len(pages), 1)
    if avg < MIN_AVG_CHARS_PER_PAGE:
        raise ScannedPDFError("Little/no text layer: likely a scanned PDF.")
    return pages
