"""Utilities for mapping retrieved contexts to readable source citations."""

from dataclasses import dataclass
import re

from app.retrieval.models import RetrievalResult


INLINE_CITATION_PATTERN = re.compile(r"[ \t]*\[(\d+)\]")


@dataclass
class SourceCitation:
    """Metadata for one numbered retrieved context."""

    citation_number: int
    document_name: str
    page_number: int
    chunk_id: str


def build_source_citations(
    results: list[RetrievalResult],
) -> list[SourceCitation]:
    """Map each numbered prompt context to its chunk metadata."""

    return [
        SourceCitation(
            citation_number=context_number,
            document_name=result.document_name,
            page_number=result.page_number,
            chunk_id=result.chunk_id,
        )
        for context_number, result in enumerate(results, start=1)
    ]


def deduplicate_page_sources(
    citations: list[SourceCitation],
) -> list[SourceCitation]:
    """Keep the first citation for each unique document page."""

    seen_pages: set[tuple[str, int]] = set()
    unique_sources: list[SourceCitation] = []

    for citation in citations:
        page_key = (citation.document_name, citation.page_number)
        if page_key in seen_pages:
            continue
        seen_pages.add(page_key)
        unique_sources.append(citation)

    return unique_sources


def extract_valid_citation_numbers(
    answer: str,
    context_count: int,
) -> list[int]:
    """Return valid citation numbers in first-appearance order."""

    if context_count < 0:
        raise ValueError("context_count must not be negative")

    citation_numbers: list[int] = []
    seen_numbers: set[int] = set()

    for match in INLINE_CITATION_PATTERN.finditer(answer):
        citation_number = int(match.group(1))
        if not 1 <= citation_number <= context_count:
            continue
        if citation_number in seen_numbers:
            continue
        seen_numbers.add(citation_number)
        citation_numbers.append(citation_number)

    return citation_numbers


def build_cited_sources(
    answer: str,
    results: list[RetrievalResult],
) -> list[SourceCitation]:
    """Build deduplicated sources for contexts cited in an answer."""

    citations = build_source_citations(results)
    citations_by_number = {
        citation.citation_number: citation for citation in citations
    }
    cited_numbers = extract_valid_citation_numbers(answer, len(results))
    cited_sources = [
        citations_by_number[citation_number]
        for citation_number in cited_numbers
    ]
    return deduplicate_page_sources(cited_sources)


def validate_inline_citations(answer: str, context_count: int) -> str:
    """Remove citation markers that do not refer to supplied contexts."""

    if context_count < 0:
        raise ValueError("context_count must not be negative")

    def replace_marker(match: re.Match[str]) -> str:
        citation_number = int(match.group(1))
        if 1 <= citation_number <= context_count:
            return match.group(0)
        return ""

    cleaned_answer = INLINE_CITATION_PATTERN.sub(replace_marker, answer)
    cleaned_answer = re.sub(r"[ \t]+([.,;:!?])", r"\1", cleaned_answer)
    return cleaned_answer.strip()
