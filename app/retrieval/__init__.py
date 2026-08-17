"""Public interface for semantic document retrieval."""

from app.retrieval.embedding_service import EmbeddingService
from app.retrieval.models import RetrievalResult
from app.retrieval.retriever import (
    DEFAULT_MIN_SCORE,
    DEFAULT_TOP_K,
    SemanticRetriever,
)
from app.retrieval.vector_store import VectorStore

__all__ = [
    "RetrievalResult",
    "EmbeddingService",
    "VectorStore",
    "SemanticRetriever",
    "DEFAULT_TOP_K",
    "DEFAULT_MIN_SCORE",
]
