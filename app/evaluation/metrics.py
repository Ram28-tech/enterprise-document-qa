"""Simple document-level semantic retrieval metrics."""


def _validate_top_k(top_k: int) -> None:
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")


def first_relevant_rank(
    expected_documents: list[str],
    retrieved_documents: list[str],
    top_k: int,
) -> int | None:
    """Return the 1-based rank of the first relevant retrieved chunk."""

    _validate_top_k(top_k)
    expected = set(expected_documents)
    if not expected:
        return None

    for rank, document_name in enumerate(
        retrieved_documents[:top_k],
        start=1,
    ):
        if document_name in expected:
            return rank

    return None


def hit_at_k(
    expected_documents: list[str],
    retrieved_documents: list[str],
    top_k: int,
) -> float:
    """Return 1 when at least one relevant chunk appears in the Top-K."""

    return float(
        first_relevant_rank(
            expected_documents,
            retrieved_documents,
            top_k,
        )
        is not None
    )


def precision_at_k(
    expected_documents: list[str],
    retrieved_documents: list[str],
    top_k: int,
) -> float:
    """Return relevant retrieved chunks divided by K."""

    _validate_top_k(top_k)
    expected = set(expected_documents)
    if not expected:
        return 0.0

    relevant_count = sum(
        document_name in expected
        for document_name in retrieved_documents[:top_k]
    )
    return relevant_count / top_k


def recall_at_k(
    expected_documents: list[str],
    retrieved_documents: list[str],
    top_k: int,
) -> float:
    """Return the fraction of unique expected documents found in Top-K."""

    _validate_top_k(top_k)
    expected = set(expected_documents)
    if not expected:
        return 0.0

    found_documents = expected.intersection(retrieved_documents[:top_k])
    return len(found_documents) / len(expected)


def reciprocal_rank(
    expected_documents: list[str],
    retrieved_documents: list[str],
    top_k: int,
) -> float:
    """Return the reciprocal rank of the first relevant retrieved chunk."""

    rank = first_relevant_rank(
        expected_documents,
        retrieved_documents,
        top_k,
    )
    return 0.0 if rank is None else 1.0 / rank
