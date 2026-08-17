"""Readable prompt construction for grounded document question answering."""

from app.retrieval.models import RetrievalResult


INSUFFICIENT_CONTEXT_MESSAGE = (
    "I could not find enough information in the provided documents."
)

SYSTEM_INSTRUCTION = """You are a document question-answering assistant.
Answer using only information supported by the supplied context.
Do not use outside knowledge or invent missing information.
If the context is insufficient, say exactly:
"I could not find enough information in the provided documents."
Keep the answer clear and concise.
For factual statements supported by context, cite its number using [1], [2], etc.
Never cite a context number that was not supplied."""


def build_context(results: list[RetrievalResult]) -> str:
    """Format retrieved chunks as clearly separated document context."""

    context_sections: list[str] = []

    for context_number, result in enumerate(results, start=1):
        context_sections.append(
            "\n".join(
                [
                    f"[Context {context_number}]",
                    f"Document: {result.document_name}",
                    f"Page: {result.page_number}",
                    f"Chunk ID: {result.chunk_id}",
                    "",
                    result.text.strip(),
                ]
            )
        )

    return "\n\n---\n\n".join(context_sections)


def build_user_prompt(
    question: str,
    results: list[RetrievalResult],
) -> str:
    """Build the user prompt from a question and retrieved evidence."""

    cleaned_question = question.strip()
    if not cleaned_question:
        raise ValueError("question must not be empty")

    context = build_context(results)
    return f"CONTEXT:\n{context}\n\nQUESTION:\n{cleaned_question}"
