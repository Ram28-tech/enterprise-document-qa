"""Inspect extracted pages and chunks for a PDF from the command line."""

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion import chunk_pages, extract_pdf_pages  # noqa: E402
from app.ingestion.chunker import (  # noqa: E402
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
)


PREVIEW_LENGTH = 200


def _preview(text: str, limit: int = PREVIEW_LENGTH) -> str:
    """Return a compact, single-line preview of chunk text."""

    compact_text = " ".join(text.split())
    if len(compact_text) <= limit:
        return compact_text
    return f"{compact_text[: limit - 3].rstrip()}..."


def main() -> None:
    """Extract and print a concise summary of a PDF's ingestion output."""

    parser = argparse.ArgumentParser(
        description="Inspect page extraction and token-budgeted PDF chunks."
    )
    parser.add_argument("pdf_path", type=Path, help="Path to the PDF to inspect")
    args = parser.parse_args()

    pages = extract_pdf_pages(args.pdf_path)
    chunks = chunk_pages(pages)

    print(f"Document: {args.pdf_path.name}")
    print(f"Extracted pages: {len(pages)}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Chunk size: {DEFAULT_CHUNK_SIZE} tokens")
    print(f"Chunk overlap: {DEFAULT_CHUNK_OVERLAP} tokens")

    if not chunks:
        print("\nNo extractable text chunks were found.")
        return

    print("\nFirst chunks:")
    for chunk in chunks[:5]:
        print(f"\nChunk ID: {chunk.chunk_id}")
        print(f"Page: {chunk.page_number}")
        print(f"Token count: {chunk.token_count}")
        print(f"Preview: {_preview(chunk.text)}")


if __name__ == "__main__":
    main()
