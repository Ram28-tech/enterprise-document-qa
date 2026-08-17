"""Sentence-aware, tokenizer-budgeted document chunking."""

from functools import lru_cache
from pathlib import Path
import re

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from app.ingestion.models import DocumentChunk, PageText


DEFAULT_CHUNK_SIZE = 220
DEFAULT_CHUNK_OVERLAP = 40
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _get_tokenizer() -> PreTrainedTokenizerBase:
    """Load the embedding model tokenizer once, on first use."""

    return AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME, use_fast=True)


def _token_count(text: str, tokenizer: PreTrainedTokenizerBase) -> int:
    """Count content tokens without model-added special tokens."""

    return len(tokenizer.encode(text, add_special_tokens=False))


def _split_sentences(text: str) -> list[str]:
    """Split on common sentence endings and paragraph boundaries."""

    parts = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    return [part.strip() for part in parts if part.strip()]


def _split_by_token_offsets(
    text: str,
    token_budget: int,
    tokenizer: PreTrainedTokenizerBase,
) -> list[str]:
    """Split text on tokenizer offsets without changing its original casing."""

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    offsets = encoded["offset_mapping"]

    if not offsets:
        return []

    pieces: list[str] = []
    start_index = 0

    while start_index < len(offsets):
        end_index = min(start_index + token_budget, len(offsets))

        while end_index > start_index:
            token_offsets = offsets[start_index:end_index]
            start_character = token_offsets[0][0]
            end_character = token_offsets[-1][1]
            piece = text[start_character:end_character].strip()

            if not piece or _token_count(piece, tokenizer) <= token_budget:
                break
            end_index -= 1

        if end_index == start_index:
            raise ValueError("Unable to split text within the requested token budget")

        if piece:
            pieces.append(piece)
        start_index = end_index

    return pieces


def _sentence_segments(
    text: str,
    chunk_size: int,
    tokenizer: PreTrainedTokenizerBase,
) -> list[str]:
    """Return sentence-sized segments that each fit the token budget."""

    segments: list[str] = []
    for sentence in _split_sentences(text):
        if _token_count(sentence, tokenizer) <= chunk_size:
            segments.append(sentence)
        else:
            segments.extend(
                _split_by_token_offsets(sentence, chunk_size, tokenizer)
            )
    return segments


def _join_parts(parts: list[str]) -> str:
    """Join non-empty text parts with readable spacing."""

    return " ".join(part.strip() for part in parts if part.strip()).strip()


def _overlap_text(
    previous_parts: list[str],
    overlap_budget: int,
    tokenizer: PreTrainedTokenizerBase,
) -> str:
    """Build a suffix overlap, preferring complete sentence segments."""

    if overlap_budget <= 0:
        return ""
    if not previous_parts:
        return ""

    selected: list[str] = []

    for part in reversed(previous_parts):
        candidate = _join_parts([part, *selected])
        if _token_count(candidate, tokenizer) <= overlap_budget:
            selected.insert(0, part)
        else:
            break

    if selected:
        return _join_parts(selected)

    trailing_part = previous_parts[-1]
    encoded = tokenizer(
        trailing_part,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    offsets = encoded["offset_mapping"]
    if not offsets:
        return ""

    trailing_offsets = offsets[-overlap_budget:]
    return trailing_part[trailing_offsets[0][0] : trailing_offsets[-1][1]].strip()


def _page_chunks(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    tokenizer: PreTrainedTokenizerBase,
) -> list[str]:
    """Chunk one page without allowing content to cross page boundaries."""

    segments = _sentence_segments(text, chunk_size, tokenizer)
    if not segments:
        return []

    chunks: list[str] = []
    current_parts: list[str] = []

    for segment in segments:
        candidate = _join_parts([*current_parts, segment])
        if _token_count(candidate, tokenizer) <= chunk_size:
            current_parts.append(segment)
            continue

        completed_chunk = _join_parts(current_parts)
        if not completed_chunk:
            raise ValueError("Unable to create a non-empty chunk within the token budget")
        chunks.append(completed_chunk)

        segment_token_count = _token_count(segment, tokenizer)
        overlap_budget = min(chunk_overlap, chunk_size - segment_token_count)
        overlap = _overlap_text(current_parts, overlap_budget, tokenizer)
        next_parts = [overlap, segment] if overlap else [segment]

        # Tokenization at a newly joined boundary can occasionally add a token.
        while overlap and _token_count(_join_parts(next_parts), tokenizer) > chunk_size:
            overlap_budget -= 1
            overlap = _overlap_text(current_parts, overlap_budget, tokenizer)
            next_parts = [overlap, segment] if overlap else [segment]

        current_parts = next_parts

    final_chunk = _join_parts(current_parts)
    if final_chunk:
        chunks.append(final_chunk)

    return chunks


def _document_id(document_name: str) -> str:
    """Create a readable identifier from a document filename."""

    filename_stem = Path(document_name).stem.lower()
    sanitized = re.sub(r"[^a-z0-9]+", "-", filename_stem).strip("-")
    return sanitized or "document"


def chunk_pages(
    pages: list[PageText],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    """Create sentence-aware, page-local chunks within a tokenizer budget."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be 0 or greater")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    if not pages:
        return []

    tokenizer = _get_tokenizer()
    document_chunks: list[DocumentChunk] = []

    for page in pages:
        page_chunks = _page_chunks(
            page.text,
            chunk_size,
            chunk_overlap,
            tokenizer,
        )
        document_id = _document_id(page.document_name)

        for chunk_index, chunk_text in enumerate(page_chunks, start=1):
            token_count = _token_count(chunk_text, tokenizer)
            if token_count > chunk_size:
                raise ValueError("Generated chunk exceeds the requested token budget")

            document_chunks.append(
                DocumentChunk(
                    chunk_id=(
                        f"{document_id}-p{page.page_number}-c{chunk_index}"
                    ),
                    document_name=page.document_name,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    token_count=token_count,
                    category=page.category,
                )
            )

    return document_chunks
