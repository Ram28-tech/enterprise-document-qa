"""Public interface for semantic retrieval evaluation."""

from app.evaluation.evaluator import RetrievalEvaluator
from app.evaluation.metrics import (
    first_relevant_rank,
    hit_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.evaluation.models import (
    EvaluationCase,
    EvaluationSummary,
    QueryEvaluation,
)

__all__ = [
    "EvaluationCase",
    "QueryEvaluation",
    "EvaluationSummary",
    "RetrievalEvaluator",
    "first_relevant_rank",
    "hit_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
]
