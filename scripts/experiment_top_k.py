"""Compare raw semantic retrieval metrics across several Top-K values."""

import csv
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation import EvaluationSummary, RetrievalEvaluator  # noqa: E402
from app.retrieval import EmbeddingService, SemanticRetriever, VectorStore  # noqa: E402
from app.retrieval.vector_store import (  # noqa: E402
    COLLECTION_NAME,
    DEFAULT_QDRANT_PATH,
)


TOP_K_VALUES = (1, 3, 5, 8)
QUESTIONS_PATH = PROJECT_ROOT / "data" / "evaluation" / "retrieval_questions.json"
RESULTS_DIRECTORY = PROJECT_ROOT / "data" / "evaluation" / "results"


def _create_evaluator() -> RetrievalEvaluator:
    """Open the existing vector collection for read-only experiments."""

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


def _save_comparison(summaries: list[EvaluationSummary]) -> Path:
    """Save the Top-K comparison as CSV."""

    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIRECTORY / "top_k_comparison.csv"

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["top_k", "hit_rate", "mean_precision", "mean_recall", "mrr"]
        )
        for summary in summaries:
            writer.writerow(
                [
                    summary.top_k,
                    f"{summary.hit_rate:.6f}",
                    f"{summary.mean_precision:.6f}",
                    f"{summary.mean_recall:.6f}",
                    f"{summary.mrr:.6f}",
                ]
            )

    return output_path


def main() -> None:
    """Evaluate and compare K values without selecting a winner."""

    evaluator = _create_evaluator()
    cases = evaluator.load_cases(QUESTIONS_PATH)
    summaries = [
        evaluator.evaluate(cases, top_k=top_k)[1]
        for top_k in TOP_K_VALUES
    ]
    output_path = _save_comparison(summaries)

    print("TOP-K RETRIEVAL COMPARISON\n")
    print("K   Hit Rate   Precision   Recall   MRR")
    for summary in summaries:
        print(
            f"{summary.top_k:<3} "
            f"{summary.hit_rate:<10.4f} "
            f"{summary.mean_precision:<11.4f} "
            f"{summary.mean_recall:<8.4f} "
            f"{summary.mrr:.4f}"
        )

    print(
        "\nHigher K may improve hit rate and recall, while it may reduce "
        "precision. MRR shows how early the first relevant result appears."
    )
    print(f"Results CSV: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
