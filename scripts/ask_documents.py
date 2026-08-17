"""Ask one grounded question using the existing local document index."""

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.generation import GeminiService, RAGService  # noqa: E402
from app.retrieval import (  # noqa: E402
    DEFAULT_MIN_SCORE,
    DEFAULT_TOP_K,
    EmbeddingService,
    SemanticRetriever,
    VectorStore,
)
from app.retrieval.vector_store import (  # noqa: E402
    COLLECTION_NAME,
    DEFAULT_QDRANT_PATH,
)


PREVIEW_LENGTH = 300


def _preview(text: str, limit: int = PREVIEW_LENGTH) -> str:
    """Return a compact chunk preview for terminal output."""

    compact_text = " ".join(text.split())
    if len(compact_text) <= limit:
        return compact_text
    return f"{compact_text[: limit - 3].rstrip()}..."


def main() -> None:
    """Retrieve evidence and generate one grounded document answer."""

    parser = argparse.ArgumentParser(
        description="Ask a grounded question using the existing Qdrant index."
    )
    parser.add_argument("question", help="Question to answer from indexed documents")
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of chunks to retrieve (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_MIN_SCORE,
        help=f"Minimum similarity score (default: {DEFAULT_MIN_SCORE:.2f})",
    )
    parser.add_argument(
        "--category",
        help="Require an exact category match",
    )
    parser.add_argument(
        "--document",
        dest="document_name",
        help="Require an exact document filename match",
    )
    args = parser.parse_args()

    qdrant_path = PROJECT_ROOT / DEFAULT_QDRANT_PATH
    if not qdrant_path.exists():
        raise FileNotFoundError(
            f"Qdrant storage was not found at '{qdrant_path}'. "
            "Run scripts/index_documents.py first."
        )

    embedding_service = EmbeddingService()
    vector_store = VectorStore(
        vector_size=embedding_service.get_dimension(),
        path=qdrant_path,
        collection_name=COLLECTION_NAME,
        create_if_missing=False,
    )

    if vector_store.count() == 0:
        raise RuntimeError(
            "The Qdrant collection contains no indexed vectors. "
            "Run scripts/index_documents.py first."
        )

    retriever = SemanticRetriever(embedding_service, vector_store)
    gemini_service = GeminiService()
    rag_service = RAGService(retriever, gemini_service)
    response = rag_service.ask(
        args.question,
        top_k=args.top_k,
        min_score=args.min_score,
        category=args.category,
        document_name=args.document_name,
    )

    print("QUESTION:")
    print(response.question)
    print("\nANSWER:")
    print(response.answer)
    print("\nSOURCES:")

    if response.sources:
        for source in response.sources:
            print(
                f"[{source.citation_number}] {source.document_name} "
                f"— Page {source.page_number}"
            )
    else:
        print("None")

    print("\nRETRIEVAL DEBUG:")

    if not response.retrieved_chunks:
        print("No relevant chunks passed the retrieval threshold.")
        return

    for result_number, result in enumerate(response.retrieved_chunks, start=1):
        print(f"\nResult {result_number}")
        print(f"Score: {result.score:.4f}")
        print(f"Document: {result.document_name}")
        print(f"Page: {result.page_number}")
        print(f"Chunk ID: {result.chunk_id}")
        print(f"Preview: {_preview(result.text)}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
