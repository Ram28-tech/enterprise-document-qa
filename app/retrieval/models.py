"""Data models returned by semantic retrieval."""

from dataclasses import dataclass


@dataclass
class RetrievalResult:
    """One document chunk returned from a semantic similarity search."""

    chunk_id: str
    document_name: str
    page_number: int
    chunk_index: int
    text: str
    token_count: int
    score: float
    category: str | None = None
