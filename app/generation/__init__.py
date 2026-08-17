"""Public interface for grounded document answer generation."""

from app.generation.citation_builder import (
    SourceCitation,
    build_source_citations,
    deduplicate_page_sources,
    validate_inline_citations,
)
from app.generation.llm_service import GeminiService
from app.generation.models import RAGResponse
from app.generation.prompt_builder import (
    INSUFFICIENT_CONTEXT_MESSAGE,
    SYSTEM_INSTRUCTION,
    build_context,
    build_user_prompt,
)
from app.generation.rag_service import RAGService

__all__ = [
    "RAGResponse",
    "SourceCitation",
    "GeminiService",
    "RAGService",
    "INSUFFICIENT_CONTEXT_MESSAGE",
    "SYSTEM_INSTRUCTION",
    "build_context",
    "build_user_prompt",
    "build_source_citations",
    "deduplicate_page_sources",
    "validate_inline_citations",
]
