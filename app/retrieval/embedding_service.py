"""SentenceTransformer embedding generation for documents and queries."""

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 32


@lru_cache(maxsize=None)
def _load_model(model_name: str) -> SentenceTransformer:
    """Load and cache one SentenceTransformer instance per model name."""

    return SentenceTransformer(model_name)


class EmbeddingService:
    """Generate normalized embeddings without vector-store responsibilities."""

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name must not be empty")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        self._model_name = model_name
        self._batch_size = batch_size
        self._model = _load_model(model_name)

    @property
    def model_name(self) -> str:
        """Return the configured SentenceTransformer model name."""

        return self._model_name

    def get_dimension(self) -> int:
        """Return the number of values in each generated embedding."""

        dimension = self._model.get_embedding_dimension()
        if dimension is None:
            raise RuntimeError("The embedding model did not report its dimension")
        return int(dimension)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Generate normalized embeddings for document chunks in one batch call."""

        if not texts:
            raise ValueError("texts must contain at least one document")
        if any(not text.strip() for text in texts):
            raise ValueError("document texts must not be empty")

        embeddings = self._model.encode_document(
            texts,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embedding_array = np.asarray(embeddings, dtype=np.float32)

        if embedding_array.ndim != 2:
            raise RuntimeError("Document embedding output must be a 2D array")

        return embedding_array

    def embed_query(self, query: str) -> np.ndarray:
        """Generate one normalized embedding for a non-empty query."""

        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("query must not be empty")

        embedding = self._model.encode_query(
            cleaned_query,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embedding_array = np.asarray(embedding, dtype=np.float32)

        if embedding_array.ndim != 1:
            raise RuntimeError("Query embedding output must be a 1D array")

        return embedding_array
