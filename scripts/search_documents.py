"""Inspect top semantic matches from the local document collection."""

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.embedding_service import EmbeddingService  # noqa: E402
from app.retrieval.retriever import (  # noqa: E402
    DEFAULT_MIN_SCORE,
    DEFAULT_TOP_K,
    SemanticRetriever,
)
from app.retrieval.vector_store import (  # noqa: E402
    COLLECTION_NAME,
    DEFAULT_QDRANT_PATH,
    VectorStore,
)


PREVIEW_LENGTH = 500


def _preview(text: str, limit: int = PREVIEW_LENGTH) -> str:
    """Return a readable preview without flooding terminal output."""

    compact_text = " ".join(text.split())
    if len(compact_text) <= limit:
        return compact_text
    return f"{compact_text[: limit - 3].rstrip()}..."


def main() -> None:
    """Embed a question and print its highest-scoring stored chunks."""

    parser = argparse.ArgumentParser(
        description="Search the local enterprise document collection."
    )
    parser.add_argument("query", help="Question or text to search for")
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of results to return (default: {DEFAULT_TOP_K})",
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

    embedding_service = EmbeddingService()
    vector_store = VectorStore(
        vector_size=embedding_service.get_dimension(),
        path=PROJECT_ROOT / DEFAULT_QDRANT_PATH,
        collection_name=COLLECTION_NAME,
        create_if_missing=False,
    )
    retriever = SemanticRetriever(embedding_service, vector_store)
    results = retriever.retrieve(
        args.query,
        top_k=args.top_k,
        min_score=args.min_score,
        category=args.category,
        document_name=args.document_name,
    )

    print("QUERY:")
    print(args.query)
    print("\nACTIVE FILTERS:")
    print(f"Category: {args.category or 'Any'}")
    print(f"Document: {args.document_name or 'Any'}")
    print(f"Minimum score: {args.min_score:.2f}")
    print("\nACCEPTED RESULTS:")

    if not results:
        print("No relevant chunks passed the retrieval threshold.")
        return

    for result_number, result in enumerate(results, start=1):
        print(f"\nResult {result_number}")
        print(f"Score: {result.score:.4f}")
        print(f"Document: {result.document_name}")
        print(f"Page: {result.page_number}")
        print(f"Chunk ID: {result.chunk_id}")
        print("Text:")
        print(_preview(result.text))


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
