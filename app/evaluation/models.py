"""Data models for document-level retrieval evaluation."""

from dataclasses import dataclass


@dataclass
class EvaluationCase:
    """One retrieval question and its expected source documents."""

    id: str
    question: str
    expected_documents: list[str]
    type: str


@dataclass
class QueryEvaluation:
    """Document-level retrieval metrics for one supported question."""

    id: str
    question: str
    expected_documents: list[str]
    retrieved_documents: list[str]
    first_relevant_rank: int | None
    hit: float
    precision: float
    recall: float
    reciprocal_rank: float


@dataclass
class EvaluationSummary:
    """Aggregate document-level metrics for supported questions."""

    top_k: int
    query_count: int
    hit_rate: float
    mean_precision: float
    mean_recall: float
    mrr: float
