"""FastAPI backend for the Enterprise Document QA application."""

from functools import lru_cache
from pathlib import Path
from threading import RLock

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from app.config import DEFAULT_MIN_SCORE, DEFAULT_TOP_K
from app.generation import GeminiService, RAGService
from app.ingestion.indexing_service import IndexingService
from app.retrieval import EmbeddingService, SemanticRetriever, VectorStore
from app.retrieval.vector_store import COLLECTION_NAME, DEFAULT_QDRANT_PATH


PROJECT_ROOT = Path(__file__).resolve().parent
DOCUMENTS_DIRECTORY = PROJECT_ROOT / "data" / "documents"
QDRANT_PATH = PROJECT_ROOT / DEFAULT_QDRANT_PATH
VECTOR_STORE_LOCK = RLock()


app = FastAPI(
    title="Enterprise Document QA API",
    description="RAG-based document question answering API",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    """Parameters for one independent grounded document question."""

    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=50)
    min_score: float = Field(default=DEFAULT_MIN_SCORE, ge=0.0, le=1.0)
    category: str | None = None
    document_name: str | None = None


class SourceResponse(BaseModel):
    """One source cited by the generated answer."""

    citation_number: int
    document_name: str
    page_number: int


class RetrievalResponse(BaseModel):
    """Compact retrieval metadata for demo/debug display."""

    score: float
    document_name: str
    page_number: int
    chunk_id: str


class QueryResponse(BaseModel):
    """Grounded answer plus cited sources and compact retrieval details."""

    question: str
    answer: str
    sources: list[SourceResponse]
    retrieval: list[RetrievalResponse]


@lru_cache(maxsize=1)
def _get_embedding_service() -> EmbeddingService:
    """Load and reuse the existing embedding model within the API process."""

    return EmbeddingService()


def _open_existing_vector_store(
    embedding_service: EmbeddingService,
) -> VectorStore:
    """Open the existing collection without creating or resetting it."""

    if not QDRANT_PATH.exists():
        raise FileNotFoundError("Document index storage is missing")

    return VectorStore(
        vector_size=embedding_service.get_dimension(),
        path=QDRANT_PATH,
        collection_name=COLLECTION_NAME,
        create_if_missing=False,
    )


def _pdf_filenames() -> list[str]:
    """Return source PDF filenames in stable order."""

    if not DOCUMENTS_DIRECTORY.exists():
        return []
    return sorted(
        path.name
        for path in DOCUMENTS_DIRECTORY.glob("*.pdf")
        if path.is_file()
    )


@app.get("/health")
def health() -> dict[str, str | int]:
    """Report API availability and the current indexed vector count."""

    if not QDRANT_PATH.exists():
        return {"status": "ok", "indexed_vectors": 0}

    try:
        with VECTOR_STORE_LOCK:
            client = QdrantClient(path=str(QDRANT_PATH))
            try:
                vector_count = (
                    int(
                        client.count(
                            collection_name=COLLECTION_NAME,
                            exact=True,
                        ).count
                    )
                    if client.collection_exists(COLLECTION_NAME)
                    else 0
                )
            finally:
                client.close()
        return {"status": "ok", "indexed_vectors": vector_count}
    except FileNotFoundError:
        return {"status": "ok", "indexed_vectors": 0}
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(
            status_code=503,
            detail="The document index is currently unavailable.",
        ) from exc


@app.get("/documents")
def list_documents() -> dict[str, list[str]]:
    """List the PDF files available as indexing sources."""

    return {"documents": _pdf_filenames()}


@app.post("/documents/upload")
async def upload_documents(
    files: list[UploadFile] = File(...),
) -> dict[str, list[str]]:
    """Validate and save PDF uploads without automatically indexing them."""

    if not files:
        raise HTTPException(status_code=400, detail="Select at least one PDF file.")

    validated_uploads: list[tuple[str, bytes]] = []
    seen_filenames: set[str] = set()

    for uploaded_file in files:
        safe_filename = Path(uploaded_file.filename or "").name
        content_type = (uploaded_file.content_type or "").lower()

        if (
            not safe_filename
            or Path(safe_filename).suffix.lower() != ".pdf"
            or content_type not in {"application/pdf", "application/x-pdf"}
        ):
            raise HTTPException(
                status_code=415,
                detail="Only PDF files are supported.",
            )
        if safe_filename.casefold() in seen_filenames:
            raise HTTPException(
                status_code=400,
                detail=f"Duplicate upload filename: {safe_filename}",
            )

        content = await uploaded_file.read()
        await uploaded_file.close()
        if b"%PDF-" not in content[:1024]:
            raise HTTPException(
                status_code=415,
                detail=f"'{safe_filename}' is not a valid PDF upload.",
            )

        seen_filenames.add(safe_filename.casefold())
        validated_uploads.append((safe_filename, content))

    DOCUMENTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    saved_filenames: list[str] = []

    try:
        for safe_filename, content in validated_uploads:
            destination = DOCUMENTS_DIRECTORY / safe_filename
            destination.write_bytes(content)
            saved_filenames.append(safe_filename)
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail="The server could not save the uploaded PDFs.",
        ) from exc

    return {"filenames": saved_filenames}


@app.post("/documents/index")
def index_documents() -> dict[str, int]:
    """Rebuild the existing document collection from source PDFs."""

    try:
        with VECTOR_STORE_LOCK:
            summary = IndexingService(
                qdrant_path=QDRANT_PATH,
            ).index_documents(DOCUMENTS_DIRECTORY, reset=True)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="Indexing failed. Verify that the uploaded PDFs are readable.",
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=500,
            detail="Document indexing failed. Please try again.",
        ) from exc

    if summary.pdf_count == 0:
        raise HTTPException(
            status_code=400,
            detail="No PDF documents are available for indexing.",
        )
    if summary.chunk_count == 0:
        raise HTTPException(
            status_code=422,
            detail="The PDFs did not contain extractable text to index.",
        )

    return {
        "pdf_count": summary.pdf_count,
        "extracted_pages": summary.extracted_page_count,
        "chunks": summary.chunk_count,
        "vectors": summary.vector_count,
    }


@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest) -> QueryResponse:
    """Run one independent question through the existing RAG service."""

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be empty.")

    try:
        with VECTOR_STORE_LOCK:
            if not QDRANT_PATH.exists():
                raise FileNotFoundError("Document index storage is missing")
            embedding_service = _get_embedding_service()
            vector_store = _open_existing_vector_store(embedding_service)
            try:
                if vector_store.count() == 0:
                    raise FileNotFoundError("Document index is empty")

                retriever = SemanticRetriever(embedding_service, vector_store)
                try:
                    gemini_service = GeminiService()
                except ValueError as exc:
                    raise HTTPException(
                        status_code=503,
                        detail=(
                            "The Gemini service is not configured on the server."
                        ),
                    ) from exc

                response = RAGService(retriever, gemini_service).ask(
                    question,
                    top_k=request.top_k,
                    min_score=request.min_score,
                    category=request.category or None,
                    document_name=request.document_name or None,
                )
            finally:
                vector_store.client.close()
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="The document index is unavailable. Index documents first.",
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail="The answer service could not complete the request.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="The query could not be processed with the supplied settings.",
        ) from exc
    return QueryResponse(
        question=response.question,
        answer=response.answer,
        sources=[
            SourceResponse(
                citation_number=source.citation_number,
                document_name=source.document_name,
                page_number=source.page_number,
            )
            for source in response.sources
        ],
        retrieval=[
            RetrievalResponse(
                score=result.score,
                document_name=result.document_name,
                page_number=result.page_number,
                chunk_id=result.chunk_id,
            )
            for result in response.retrieved_chunks
        ],
    )
