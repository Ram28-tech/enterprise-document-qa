"""Semantic retrieval orchestration."""

import math

from app.config import DEFAULT_MIN_SCORE, DEFAULT_TOP_K
from app.retrieval.embedding_service import EmbeddingService
from app.retrieval.models import RetrievalResult
from app.retrieval.vector_store import VectorStore


class SemanticRetriever:
    """Embed a query and retrieve its nearest document chunks."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
    ) -> None:
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        min_score: float | None = DEFAULT_MIN_SCORE,
        category: str | None = None,
        document_name: str | None = None,
    ) -> list[RetrievalResult]:
        """Return metadata-filtered matches that pass the score threshold."""

        if min_score is not None and (
            not math.isfinite(min_score) or not -1.0 <= min_score <= 1.0
        ):
            raise ValueError("min_score must be between -1.0 and 1.0, or None")

        query_vector = self.embedding_service.embed_query(query)
        candidates = self.vector_store.search(
            query_vector,
            top_k=top_k,
            category=category,
            document_name=document_name,
        )

        if min_score is None:
            return candidates

        return [result for result in candidates if result.score >= min_score]
