"""
PDF text extraction and text chunking utilities.
Shared by app.py and scripts/ci_eval.py to ensure consistent behaviour.
"""

from __future__ import annotations

import re
from typing import Any

import fitz  # PyMuPDF


# ---------------------------------------------------------------------------
# Config defaults — override by passing keyword args where needed
# ---------------------------------------------------------------------------
DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 150


def clean_text(text: str) -> str:
    """Collapse whitespace and strip surrounding whitespace."""
    return re.sub(r"\s+", " ", (text or "")).strip()


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """
    Split *text* into overlapping chunks of at most *chunk_size* characters.

    The overlap ensures that sentences split across chunk boundaries are still
    represented in both neighbouring chunks, reducing retrieval gaps.
    """
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def extract_pdf_pages(pdf_bytes: bytes) -> list[dict[str, Any]]:
    """
    Extract per-page text from a PDF supplied as raw bytes.

    Returns a list of dicts: ``{"page_num": int, "text": str}``.
    Pages with no extractable text are skipped.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages: list[dict[str, Any]] = []
    for i in range(len(doc)):
        txt = clean_text(doc[i].get_text("text"))
        if txt:
            pages.append({"page_num": i + 1, "text": txt})
    doc.close()
    return pages


def extract_pdf_pages_from_path(pdf_path: str) -> list[dict[str, Any]]:
    """
    Extract per-page text from a PDF at *pdf_path* on disk.

    Returns a list of dicts: ``{"page_num": int, "text": str}``.
    """
    with open(pdf_path, "rb") as fh:
        return extract_pdf_pages(fh.read())


def pdf_to_chunks(
    pdf_bytes: bytes,
    source_name: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """
    Extract text from a PDF and split it into overlapping chunks.

    Returns a list of dicts::

        {
            "id":      "<source_name>|p<page>|c<idx>",
            "text":    "<chunk text>",
            "source":  "<source_name>",
            "page":    <page_num: int>,
        }
    """
    pages = extract_pdf_pages(pdf_bytes)
    records: list[dict[str, Any]] = []
    for page in pages:
        for idx, chunk in enumerate(chunk_text(page["text"], chunk_size, overlap)):
            records.append(
                {
                    "id": f"{source_name}|p{page['page_num']}|c{idx}",
                    "text": chunk,
                    "source": source_name,
                    "page": page["page_num"],
                }
            )
    return records
