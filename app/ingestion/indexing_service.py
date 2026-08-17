"""Reusable orchestration for indexing PDF documents into Qdrant."""

from dataclasses import dataclass
from pathlib import Path

from app.ingestion.chunker import chunk_pages
from app.ingestion.models import DocumentChunk
from app.ingestion.pdf_loader import extract_pdf_pages
from app.retrieval.embedding_service import EmbeddingService
from app.retrieval.vector_store import (
    COLLECTION_NAME,
    DEFAULT_QDRANT_PATH,
    VectorStore,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class IndexingSummary:
    """Counts and storage details from one indexing run."""

    pdf_count: int
    extracted_page_count: int
    chunk_count: int
    vector_count: int
    embedding_model: str | None = None
    embedding_dimension: int | None = None
    collection_name: str | None = None
    storage_path: str | None = None


class IndexingService:
    """Extract, chunk, embed, and store a directory of PDF documents."""

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        qdrant_path: str | Path | None = None,
    ) -> None:
        self._embedding_service = embedding_service
        self.qdrant_path = Path(
            qdrant_path
            if qdrant_path is not None
            else PROJECT_ROOT / DEFAULT_QDRANT_PATH
        )

    def index_documents(
        self,
        documents_dir: Path,
        reset: bool = True,
    ) -> IndexingSummary:
        """Index every PDF below ``documents_dir`` and return run counts."""

        pdf_paths = self._find_pdfs(documents_dir)
        if not pdf_paths:
            return IndexingSummary(0, 0, 0, 0)

        extracted_page_count = 0
        all_chunks: list[DocumentChunk] = []

        for pdf_path in pdf_paths:
            pages = extract_pdf_pages(pdf_path)
            chunks = chunk_pages(pages)
            extracted_page_count += len(pages)
            all_chunks.extend(chunks)

        if not all_chunks:
            return IndexingSummary(
                pdf_count=len(pdf_paths),
                extracted_page_count=extracted_page_count,
                chunk_count=0,
                vector_count=0,
            )

        embedding_service = self._embedding_service or EmbeddingService()
        embedding_dimension = embedding_service.get_dimension()
        embeddings = embedding_service.embed_documents(
            [chunk.text for chunk in all_chunks]
        )

        vector_store = VectorStore(
            vector_size=embedding_dimension,
            path=self.qdrant_path,
            collection_name=COLLECTION_NAME,
        )
        try:
            if reset:
                vector_store.reset_collection()

            vector_store.upsert_chunks(all_chunks, embeddings)
            vector_count = vector_store.count()

            return IndexingSummary(
                pdf_count=len(pdf_paths),
                extracted_page_count=extracted_page_count,
                chunk_count=len(all_chunks),
                vector_count=vector_count,
                embedding_model=embedding_service.model_name,
                embedding_dimension=embedding_dimension,
                collection_name=vector_store.collection_name,
                storage_path=str(vector_store.path),
            )
        finally:
            vector_store.client.close()

    @staticmethod
    def _find_pdfs(directory: Path) -> list[Path]:
        """Return every PDF below a directory in a stable order."""

        if not directory.exists() or not directory.is_dir():
            return []

        return sorted(
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() == ".pdf"
        )
