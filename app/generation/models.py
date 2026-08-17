"""Data models returned by the complete RAG pipeline."""

from dataclasses import dataclass

from app.generation.citation_builder import SourceCitation
from app.retrieval.models import RetrievalResult


@dataclass
class RAGResponse:
    """A generated answer together with the chunks used as evidence."""

    question: str
    answer: str
    retrieved_chunks: list[RetrievalResult]
    sources: list[SourceCitation]
