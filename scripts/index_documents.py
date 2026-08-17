"""Index all PDFs in data/documents into persistent local Qdrant storage."""

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.indexing_service import IndexingService  # noqa: E402


DOCUMENTS_DIRECTORY = PROJECT_ROOT / "data" / "documents"


def main() -> None:
    """Extract, chunk, embed, and persist every available PDF."""

    parser = argparse.ArgumentParser(
        description="Index PDFs into the local enterprise document collection."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Recreate the Qdrant collection before indexing",
    )
    args = parser.parse_args()

    summary = IndexingService().index_documents(
        DOCUMENTS_DIRECTORY,
        reset=args.reset,
    )
    print(f"PDF files found: {summary.pdf_count}")

    if summary.pdf_count == 0:
        print(f"No PDF files found in: {DOCUMENTS_DIRECTORY}")
        return

    print(f"Extracted pages: {summary.extracted_page_count}")
    print(f"Total chunks: {summary.chunk_count}")

    if summary.chunk_count == 0:
        print("No extractable text chunks were found; nothing was indexed.")
        return

    print(f"Embedding model: {summary.embedding_model}")
    print(f"Embedding dimension: {summary.embedding_dimension}")
    print(f"Vectors stored: {summary.vector_count}")
    print(f"Qdrant collection: {summary.collection_name}")
    print(f"Qdrant storage path: {summary.storage_path}")


if __name__ == "__main__":
    main()
