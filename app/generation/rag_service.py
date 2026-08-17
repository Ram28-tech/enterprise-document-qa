"""Orchestration for retrieval-augmented document question answering."""

from app.generation.citation_builder import (
    build_cited_sources,
    validate_inline_citations,
)
from app.generation.llm_service import GeminiService
from app.generation.models import RAGResponse
from app.generation.prompt_builder import (
    INSUFFICIENT_CONTEXT_MESSAGE,
    SYSTEM_INSTRUCTION,
    build_user_prompt,
)
from app.retrieval.retriever import (
    DEFAULT_MIN_SCORE,
    DEFAULT_TOP_K,
    SemanticRetriever,
)


class RAGService:
    """Retrieve document evidence and request one grounded Gemini answer."""

    def __init__(
        self,
        retriever: SemanticRetriever,
        gemini_service: GeminiService,
    ) -> None:
        self.retriever = retriever
        self.gemini_service = gemini_service

    def ask(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        min_score: float | None = DEFAULT_MIN_SCORE,
        category: str | None = None,
        document_name: str | None = None,
    ) -> RAGResponse:
        """Answer a question using only the top retrieved document chunks."""

        cleaned_question = question.strip()
        if not cleaned_question:
            raise ValueError("question must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        retrieved_chunks = self.retriever.retrieve(
            cleaned_question,
            top_k=top_k,
            min_score=min_score,
            category=category,
            document_name=document_name,
        )

        if not retrieved_chunks:
            return RAGResponse(
                question=cleaned_question,
                answer=INSUFFICIENT_CONTEXT_MESSAGE,
                retrieved_chunks=[],
                sources=[],
            )

        prompt = build_user_prompt(cleaned_question, retrieved_chunks)
        generated_answer = self.gemini_service.generate(prompt, SYSTEM_INSTRUCTION)
        answer = validate_inline_citations(
            generated_answer,
            context_count=len(retrieved_chunks),
        )
        if not answer:
            answer = INSUFFICIENT_CONTEXT_MESSAGE

        sources = build_cited_sources(answer, retrieved_chunks)

        return RAGResponse(
            question=cleaned_question,
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            sources=sources,
        )
