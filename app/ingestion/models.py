"""Data models produced by the document ingestion pipeline."""

from dataclasses import dataclass


@dataclass
class PageText:
    """Cleaned text extracted from one PDF page."""

    document_name: str
    page_number: int
    text: str
    category: str | None = None


@dataclass
class DocumentChunk:
    """A token-budgeted section of text from a single PDF page."""

    chunk_id: str
    document_name: str
    page_number: int
    chunk_index: int
    text: str
    token_count: int
    category: str | None = None
