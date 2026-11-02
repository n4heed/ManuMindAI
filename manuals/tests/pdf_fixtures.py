"""
Builds real, temporary PDF files with PyMuPDF for ingestion tests - Task B
asked for actual files where practical rather than mocking extraction, so
these exercise the real `import fitz` / `fitz.open()` code path, not a stub.
"""

import os
import tempfile

import fitz


def make_valid_pdf(pages: list[str]) -> str:
    """A real multi-page PDF with distinct text per page. Returns a temp file path.

    Uses insert_textbox (wrapped, within a rect) rather than insert_text: a
    plain insert_text() call does not wrap and silently drops whatever does
    not fit on one line, which would make a test's "long text" fixture
    actually extract as a short, truncated string - found while writing the
    chunk-splitting regression test.
    """
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        remaining = page.insert_textbox(fitz.Rect(50, 50, 545, 790), text, fontsize=11)
        if remaining < 0:
            raise ValueError("make_valid_pdf: text does not fit in the page rect, shorten it")
    path = tempfile.mktemp(suffix=".pdf")
    doc.save(path)
    doc.close()
    return path


def make_corrupt_pdf() -> str:
    """Not a real PDF at all - garbage bytes with a .pdf extension."""
    path = tempfile.mktemp(suffix=".pdf")
    with open(path, "wb") as f:
        f.write(b"this is not a pdf file, just random bytes with a .pdf name\x00\x01\x02")
    return path


def make_encrypted_pdf(user_password: str = "secret123") -> str:
    """A real, validly-structured PDF that requires a password to open."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "This content requires a password to read.")
    path = tempfile.mktemp(suffix=".pdf")
    doc.save(path, encryption=fitz.PDF_ENCRYPT_AES_256, user_pw=user_password, owner_pw="owner-" + user_password)
    doc.close()
    return path


def make_image_only_pdf() -> str:
    """A real PDF with a page that has a picture but zero extractable text (simulates a scan)."""
    doc = fitz.open()
    page = doc.new_page()
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 50, 50))
    pix.set_rect(pix.irect, (200, 0, 0))
    page.insert_image(fitz.Rect(10, 10, 60, 60), pixmap=pix)
    path = tempfile.mktemp(suffix=".pdf")
    doc.save(path)
    doc.close()
    return path


def cleanup(*paths: str) -> None:
    for path in paths:
        if path and os.path.exists(path):
            os.remove(path)
