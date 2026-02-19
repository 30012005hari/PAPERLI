"""
PDF text extraction module.

Uses pdfplumber to pull text from every page of a research-paper PDF.
"""

from __future__ import annotations

import io
import re
import unicodedata
from typing import BinaryIO

import pdfplumber


def extract_text(pdf_file: BinaryIO) -> tuple[str, list[str]]:
    """
    Extract text from an uploaded PDF.

    Parameters
    ----------
    pdf_file : BinaryIO
        A file-like object (e.g. from Streamlit's file_uploader).

    Returns
    -------
    full_text : str
        The entire document text, cleaned.
    pages : list[str]
        Per-page text list.
    """
    pages: list[str] = []

    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            raw = page.extract_text() or ""
            cleaned = _clean_page(raw)
            pages.append(cleaned)

    full_text = "\n\n".join(pages)
    return full_text, pages


def _clean_page(text: str) -> str:
    """Basic cleaning: normalise unicode, strip repeated whitespace."""
    # Normalise unicode characters
    text = unicodedata.normalize("NFKD", text)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace on each line
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(lines)
