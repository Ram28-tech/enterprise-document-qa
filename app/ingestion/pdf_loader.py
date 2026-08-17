"""PDF page extraction for the document ingestion pipeline."""

from pathlib import Path

import pymupdf

from app.ingestion.models import PageText
from app.ingestion.text_cleaner import clean_text


def extract_pdf_pages(
    pdf_path: str | Path,
    category: str | None = None,
) -> list[PageText]:
    """Extract and clean non-empty text pages from a PDF.

    Page numbers in the returned models are 1-based. Image-only pages are
    skipped because OCR is outside the scope of this ingestion pipeline.

    Raises:
        FileNotFoundError: If the supplied path does not point to a file.
        ValueError: If the file is not a PDF or cannot be read as one.
    """

    path = Path(pdf_path)

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, received: {path.name}")

    extracted_pages: list[PageText] = []

    try:
        with pymupdf.open(path) as document:
            if document.needs_pass:
                raise ValueError(
                    f"Cannot read password-protected PDF without a password: {path}"
                )

            for page_index, page in enumerate(document):
                cleaned_text = clean_text(page.get_text("text", sort=True))

                if not cleaned_text:
                    continue

                extracted_pages.append(
                    PageText(
                        document_name=path.name,
                        page_number=page_index + 1,
                        text=cleaned_text,
                        category=category,
                    )
                )
    except ValueError:
        raise
    except (pymupdf.FileDataError, RuntimeError) as exc:
        raise ValueError(
            f"Could not read PDF '{path}'. The file may be invalid or corrupt."
        ) from exc

    return extracted_pages
