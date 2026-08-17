"""Orchestration for document-level semantic retrieval evaluation."""

import json
from pathlib import Path
from statistics import fmean

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
from app.retrieval.retriever import SemanticRetriever


class RetrievalEvaluator:
    """Evaluate raw semantic rankings without calling answer generation."""

    def __init__(self, retriever: SemanticRetriever) -> None:
        self.retriever = retriever

    @staticmethod
    def load_cases(path: str | Path) -> list[EvaluationCase]:
        """Load evaluation cases from a JSON array."""

        evaluation_path = Path(path)
        if not evaluation_path.exists() or not evaluation_path.is_file():
            raise FileNotFoundError(
                f"Evaluation questions file not found: {evaluation_path}"
            )

        with evaluation_path.open("r", encoding="utf-8") as file:
            raw_cases = json.load(file)

        if not isinstance(raw_cases, list):
            raise ValueError("Evaluation questions must be stored as a JSON array")

        cases: list[EvaluationCase] = []
        seen_ids: set[str] = set()

        for raw_case in raw_cases:
            if not isinstance(raw_case, dict):
                raise ValueError("Each evaluation case must be a JSON object")
            if not isinstance(raw_case.get("expected_documents"), list):
                raise ValueError(
                    "Each expected_documents value must be a JSON array"
                )

            try:
                case = EvaluationCase(
                    id=str(raw_case["id"]),
                    question=str(raw_case["question"]),
                    expected_documents=list(raw_case["expected_documents"]),
                    type=str(raw_case["type"]),
                )
            except (KeyError, TypeError) as exc:
                raise ValueError(
                    "Each evaluation case must contain id, question, "
                    "expected_documents, and type"
                ) from exc

            if (
                not case.id.strip()
                or not case.question.strip()
                or not case.type.strip()
            ):
                raise ValueError(
                    "Evaluation case id, question, and type must not be empty"
                )
            if case.id in seen_ids:
                raise ValueError(f"Duplicate evaluation case id: {case.id}")
            if not all(
                isinstance(document_name, str)
                and document_name.strip()
                for document_name in case.expected_documents
            ):
                raise ValueError(
                    f"Evaluation case {case.id} has an invalid document name"
                )

            seen_ids.add(case.id)
            cases.append(case)

        return cases

    def evaluate(
        self,
        cases: list[EvaluationCase],
        top_k: int,
    ) -> tuple[list[QueryEvaluation], EvaluationSummary]:
        """Evaluate supported questions using unthresholded semantic ranking."""

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        query_evaluations: list[QueryEvaluation] = []

        for case in cases:
            if not case.expected_documents:
                continue

            results = self.retriever.retrieve(
                case.question,
                top_k=top_k,
                min_score=None,
            )
            retrieved_documents = [
                result.document_name for result in results
            ]

            query_evaluations.append(
                QueryEvaluation(
                    id=case.id,
                    question=case.question,
                    expected_documents=case.expected_documents,
                    retrieved_documents=retrieved_documents,
                    first_relevant_rank=first_relevant_rank(
                        case.expected_documents,
                        retrieved_documents,
                        top_k,
                    ),
                    hit=hit_at_k(
                        case.expected_documents,
                        retrieved_documents,
                        top_k,
                    ),
                    precision=precision_at_k(
                        case.expected_documents,
                        retrieved_documents,
                        top_k,
                    ),
                    recall=recall_at_k(
                        case.expected_documents,
                        retrieved_documents,
                        top_k,
                    ),
                    reciprocal_rank=reciprocal_rank(
                        case.expected_documents,
                        retrieved_documents,
                        top_k,
                    ),
                )
            )

        summary = EvaluationSummary(
            top_k=top_k,
            query_count=len(query_evaluations),
            hit_rate=_mean_metric(query_evaluations, "hit"),
            mean_precision=_mean_metric(query_evaluations, "precision"),
            mean_recall=_mean_metric(query_evaluations, "recall"),
            mrr=_mean_metric(query_evaluations, "reciprocal_rank"),
        )

        return query_evaluations, summary


def _mean_metric(
    evaluations: list[QueryEvaluation],
    field_name: str,
) -> float:
    """Return a safe arithmetic mean for one QueryEvaluation field."""

    if not evaluations:
        return 0.0
    return fmean(
        float(getattr(evaluation, field_name))
        for evaluation in evaluations
    )
