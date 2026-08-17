"""Persistent local Qdrant storage and nearest-neighbor search."""

from pathlib import Path
import uuid

import numpy as np
from qdrant_client import QdrantClient, models

from app.ingestion.models import DocumentChunk
from app.retrieval.models import RetrievalResult


COLLECTION_NAME = "enterprise_documents"
DEFAULT_QDRANT_PATH = Path("vector_store/qdrant")


class VectorStore:
    """Store document vectors and search them using local Qdrant."""

    def __init__(
        self,
        vector_size: int,
        path: str | Path = DEFAULT_QDRANT_PATH,
        collection_name: str = COLLECTION_NAME,
        create_if_missing: bool = True,
    ) -> None:
        if vector_size <= 0:
            raise ValueError("vector_size must be greater than 0")
        if not collection_name.strip():
            raise ValueError("collection_name must not be empty")

        self.vector_size = vector_size
        self.path = Path(path)
        self.collection_name = collection_name

        if not create_if_missing and not self.path.exists():
            raise FileNotFoundError(
                f"Qdrant storage was not found at '{self.path}'. "
                "Run scripts/index_documents.py first."
            )

        self.path.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(self.path))

        if not self.collection_exists():
            if not create_if_missing:
                self.client.close()
                raise FileNotFoundError(
                    f"Qdrant collection '{self.collection_name}' does not exist. "
                    "Run scripts/index_documents.py first."
                )
            self._create_collection()

    def _create_collection(self) -> None:
        """Create the configured cosine-similarity collection."""

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    def _collection_vector_size(self) -> int:
        """Read the vector size configured in the existing collection."""

        collection = self.client.get_collection(self.collection_name)
        vector_config = collection.config.params.vectors

        if not isinstance(vector_config, models.VectorParams):
            raise ValueError("Named vector collections are not supported")

        return int(vector_config.size)

    def _validate_vector_dimension(self, dimension: int) -> None:
        """Ensure a vector matches both this store and its collection."""

        collection_size = self._collection_vector_size()

        if dimension != self.vector_size:
            raise ValueError(
                f"Embedding dimension {dimension} does not match configured "
                f"vector size {self.vector_size}"
            )
        if dimension != collection_size:
            raise ValueError(
                f"Embedding dimension {dimension} does not match existing "
                f"collection vector size {collection_size}. Reset the collection "
                "before indexing with a different embedding model."
            )

    def collection_exists(self) -> bool:
        """Return whether the configured collection currently exists."""

        return self.client.collection_exists(self.collection_name)

    def count(self) -> int:
        """Return the exact number of points stored in the collection."""

        if not self.collection_exists():
            return 0
        return int(
            self.client.count(
                collection_name=self.collection_name,
                exact=True,
            ).count
        )

    def reset_collection(self) -> None:
        """Delete and recreate the configured cosine-similarity collection."""

        if self.collection_exists():
            # Clearing first also makes reset reliable for local SQLite storage
            # on Windows, where an open file handle can delay directory removal.
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(filter=models.Filter()),
                wait=True,
            )
            self.client.delete_collection(self.collection_name)
        self._create_collection()

    def upsert_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: np.ndarray,
    ) -> None:
        """Insert or update chunks using deterministic UUID point IDs."""

        embedding_array = np.asarray(embeddings, dtype=np.float32)

        if embedding_array.ndim != 2:
            raise ValueError("embeddings must be a 2D array")
        if len(chunks) != embedding_array.shape[0]:
            raise ValueError(
                "The number of chunks must equal the number of embeddings"
            )
        if embedding_array.shape[1] != self.vector_size:
            self._validate_vector_dimension(embedding_array.shape[1])
        else:
            self._validate_vector_dimension(self.vector_size)

        if not chunks:
            return

        points: list[models.PointStruct] = []
        for chunk, vector in zip(chunks, embedding_array, strict=True):
            point_id = uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"enterprise-document-qa:{chunk.chunk_id}",
            )
            payload = {
                "chunk_id": chunk.chunk_id,
                "document_name": chunk.document_name,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "token_count": chunk.token_count,
                "category": chunk.category,
            }
            points.append(
                models.PointStruct(
                    id=str(point_id),
                    vector=vector.tolist(),
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        category: str | None = None,
        document_name: str | None = None,
    ) -> list[RetrievalResult]:
        """Return nearest chunks within optional exact metadata filters."""

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        vector = np.asarray(query_vector, dtype=np.float32)
        if vector.ndim != 1:
            raise ValueError("query_vector must be a 1D array")

        self._validate_vector_dimension(vector.shape[0])

        filter_conditions: list[models.FieldCondition] = []
        if category is not None:
            filter_conditions.append(
                models.FieldCondition(
                    key="category",
                    match=models.MatchValue(value=category),
                )
            )
        if document_name is not None:
            filter_conditions.append(
                models.FieldCondition(
                    key="document_name",
                    match=models.MatchValue(value=document_name),
                )
            )

        metadata_filter = (
            models.Filter(must=filter_conditions) if filter_conditions else None
        )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector.tolist(),
            query_filter=metadata_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        results: list[RetrievalResult] = []
        for point in response.points:
            payload = point.payload or {}
            try:
                category = payload.get("category")
                results.append(
                    RetrievalResult(
                        chunk_id=str(payload["chunk_id"]),
                        document_name=str(payload["document_name"]),
                        page_number=int(payload["page_number"]),
                        chunk_index=int(payload["chunk_index"]),
                        text=str(payload["text"]),
                        token_count=int(payload["token_count"]),
                        score=float(point.score),
                        category=None if category is None else str(category),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Stored point {point.id} has an invalid chunk payload"
                ) from exc

        return results
