"""Evaluate raw semantic retrieval rankings at one Top-K value."""

import argparse
from dataclasses import asdict
import csv
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation import RetrievalEvaluator  # noqa: E402
from app.evaluation.models import (  # noqa: E402
    EvaluationSummary,
    QueryEvaluation,
)
from app.retrieval import EmbeddingService, SemanticRetriever, VectorStore  # noqa: E402
from app.retrieval.vector_store import (  # noqa: E402
    COLLECTION_NAME,
    DEFAULT_QDRANT_PATH,
)


QUESTIONS_PATH = PROJECT_ROOT / "data" / "evaluation" / "retrieval_questions.json"
RESULTS_DIRECTORY = PROJECT_ROOT / "data" / "evaluation" / "results"


def _create_evaluator() -> RetrievalEvaluator:
    """Open the existing index and create a retrieval-only evaluator."""

    embedding_service = EmbeddingService()
    vector_store = VectorStore(
        vector_size=embedding_service.get_dimension(),
        path=PROJECT_ROOT / DEFAULT_QDRANT_PATH,
        collection_name=COLLECTION_NAME,
        create_if_missing=False,
    )
    if vector_store.count() == 0:
        raise RuntimeError(
            "The Qdrant collection contains no indexed vectors. "
            "Run scripts/index_documents.py first."
        )

    return RetrievalEvaluator(
        SemanticRetriever(embedding_service, vector_store)
    )


def _save_results(
    evaluations: list[QueryEvaluation],
    summary: EvaluationSummary,
) -> tuple[Path, Path]:
    """Save per-query CSV data and one JSON summary."""

    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIRECTORY / f"retrieval_k{summary.top_k}.csv"
    summary_path = (
        RESULTS_DIRECTORY / f"retrieval_k{summary.top_k}_summary.json"
    )

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "id",
                "question",
                "expected_documents",
                "retrieved_documents",
                "first_relevant_rank",
                "hit",
                "precision",
                "recall",
                "reciprocal_rank",
            ]
        )
        for evaluation in evaluations:
            writer.writerow(
                [
                    evaluation.id,
                    evaluation.question,
                    "|".join(evaluation.expected_documents),
                    "|".join(evaluation.retrieved_documents),
                    evaluation.first_relevant_rank or "",
                    f"{evaluation.hit:.6f}",
                    f"{evaluation.precision:.6f}",
                    f"{evaluation.recall:.6f}",
                    f"{evaluation.reciprocal_rank:.6f}",
                ]
            )

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(asdict(summary), file, indent=2)
        file.write("\n")

    return csv_path, summary_path


def main() -> None:
    """Run and print one retrieval ranking evaluation."""

    parser = argparse.ArgumentParser(
        description="Evaluate raw semantic retrieval at a selected Top-K."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieved chunks per question (default: 5)",
    )
    args = parser.parse_args()

    evaluator = _create_evaluator()
    cases = evaluator.load_cases(QUESTIONS_PATH)
    evaluations, summary = evaluator.evaluate(cases, top_k=args.top_k)
    csv_path, summary_path = _save_results(evaluations, summary)

    print("RETRIEVAL EVALUATION\n")
    print(f"Top-K: {summary.top_k}")
    print(f"Supported questions: {summary.query_count}\n")
    print(f"Hit Rate@{summary.top_k}: {summary.hit_rate:.4f}")
    print(f"Mean Precision@{summary.top_k}: {summary.mean_precision:.4f}")
    print(f"Mean Recall@{summary.top_k}: {summary.mean_recall:.4f}")
    print(f"MRR: {summary.mrr:.4f}")

    print("\nPER-QUESTION RESULTS:")
    for evaluation in evaluations:
        rank = evaluation.first_relevant_rank or "None"
        print(f"\n{evaluation.id}")
        print(f"Hit: {evaluation.hit:.1f}")
        print(f"First relevant rank: {rank}")
        print(f"Expected: {', '.join(evaluation.expected_documents)}")
        print(f"Retrieved: {', '.join(evaluation.retrieved_documents)}")

    print(f"\nPer-query CSV: {csv_path}")
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
