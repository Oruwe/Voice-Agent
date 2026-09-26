"""
Parse raw document bytes into text chunks suitable for Moss/Qdrant upsert.

Supports PDF, DOCX, and plain text. A stable hash of the full text is used
as the doc prefix so re-uploading the same file produces identical chunk IDs
(idempotent upsert).
"""
from __future__ import annotations

import hashlib
import io


def parse_document(content: bytes, content_type: str) -> str:
    """Extract plain text from document bytes. Raises ValueError for unsupported types."""
    ct = (content_type or "").lower()
    if "pdf" in ct:
        return _parse_pdf(content)
    if "word" in ct or "docx" in ct or "openxmlformats" in ct:
        return _parse_docx(content)
    # Plain text / fallback
    return content.decode("utf-8", errors="replace")


def _parse_pdf(content: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(content))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _parse_docx(content: bytes) -> str:
    from docx import Document

    doc = Document(io.BytesIO(content))
    return "\n".join(para.text for para in doc.paragraphs)


def chunk_text(
    text: str,
    *,
    doc_name: str = "document",
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[dict]:
    """Split text into overlapping fixed-size chunks.

    Returns a list of dicts with 'id', 'text', and 'metadata' keys ready for
    MossContextProvider.upsert_context() and QdrantProvider.upsert().
    """
    text = text.strip()
    if not text:
        return []

    doc_hash = hashlib.sha256(text.encode()).hexdigest()[:12]
    chunks: list[dict] = []
    start = 0
    idx = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({
                "id": f"{doc_hash}:{idx}",
                "text": chunk,
                "metadata": {
                    "source": doc_name,
                    "chunk": idx,
                    "doc_hash": doc_hash,
                },
            })
            idx += 1
        start = end - overlap if end < len(text) else len(text)

    return chunks
