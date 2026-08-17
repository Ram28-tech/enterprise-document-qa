"""Compare score thresholds on supported and unsupported questions."""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation import EvaluationCase, RetrievalEvaluator  # noqa: E402
from app.retrieval import EmbeddingService, SemanticRetriever, VectorStore  # noqa: E402
from app.retrieval.vector_store import (  # noqa: E402
    COLLECTION_NAME,
    DEFAULT_QDRANT_PATH,
)


THRESHOLDS = (0.40, 0.50, 0.60)
QUESTIONS_PATH = PROJECT_ROOT / "data" / "evaluation" / "retrieval_questions.json"
RESULTS_DIRECTORY = PROJECT_ROOT / "data" / "evaluation" / "results"


@dataclass
class ThresholdResult:
    """Accept/reject decision rates for one similarity threshold."""

    threshold: float
    supported_success_rate: float
    unsupported_rejection_rate: float
    overall_decision_accuracy: float


def _create_retriever() -> SemanticRetriever:
    """Open the existing vector collection without modifying it."""

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

    return SemanticRetriever(embedding_service, vector_store)


def _evaluate_threshold(
    retriever: SemanticRetriever,
    cases: list[EvaluationCase],
    threshold: float,
    top_k: int,
) -> ThresholdResult:
    """Evaluate supported acceptance and unsupported rejection decisions."""

    supported_count = 0
    supported_successes = 0
    unsupported_count = 0
    unsupported_rejections = 0

    for case in cases:
        results = retriever.retrieve(
            case.question,
            top_k=top_k,
            min_score=threshold,
        )

        if case.expected_documents:
            supported_count += 1
            expected = set(case.expected_documents)
            if results and any(
                result.document_name in expected for result in results
            ):
                supported_successes += 1
        else:
            unsupported_count += 1
            if not results:
                unsupported_rejections += 1

    supported_rate = (
        supported_successes / supported_count if supported_count else 0.0
    )
    unsupported_rate = (
        unsupported_rejections / unsupported_count
        if unsupported_count
        else 0.0
    )
    total_count = supported_count + unsupported_count
    overall_accuracy = (
        (supported_successes + unsupported_rejections) / total_count
        if total_count
        else 0.0
    )

    return ThresholdResult(
        threshold=threshold,
        supported_success_rate=supported_rate,
        unsupported_rejection_rate=unsupported_rate,
        overall_decision_accuracy=overall_accuracy,
    )


def _save_comparison(results: list[ThresholdResult]) -> Path:
    """Save threshold decision rates as CSV."""

    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIRECTORY / "threshold_comparison.csv"

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "threshold",
                "supported_success_rate",
                "unsupported_rejection_rate",
                "overall_decision_accuracy",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    f"{result.threshold:.2f}",
                    f"{result.supported_success_rate:.6f}",
                    f"{result.unsupported_rejection_rate:.6f}",
                    f"{result.overall_decision_accuracy:.6f}",
                ]
            )

    return output_path


def main() -> None:
    """Run the supported/unsupported threshold experiment."""

    parser = argparse.ArgumentParser(
        description="Compare retrieval score threshold decisions."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Candidate results per question before thresholding (default: 5)",
    )
    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    retriever = _create_retriever()
    cases = RetrievalEvaluator.load_cases(QUESTIONS_PATH)
    results = [
        _evaluate_threshold(
            retriever,
            cases,
            threshold,
            top_k=args.top_k,
        )
        for threshold in THRESHOLDS
    ]
    output_path = _save_comparison(results)

    print("RETRIEVAL THRESHOLD COMPARISON\n")
    print(f"Top-K candidates: {args.top_k}\n")
    print("Threshold   Supported Success   Unsupported Rejection   Overall")
    for result in results:
        print(
            f"{result.threshold:<11.2f} "
            f"{result.supported_success_rate:<19.4f} "
            f"{result.unsupported_rejection_rate:<23.4f} "
            f"{result.overall_decision_accuracy:.4f}"
        )

    print("\nThreshold values are similarity-score cutoffs, not probabilities.")
    print(f"Results CSV: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
