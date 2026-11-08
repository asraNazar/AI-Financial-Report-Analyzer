
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any


def _validate_k(k: int) -> None:
    """Validate the requested number of top results."""
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be an integer of at least 1")


def precision_at_k(
    retrieved_indices: Sequence[int],
    relevant_indices: Iterable[int],
    k: int = 3,
) -> float:
    """
    Calculate Precision@K.

    retrieved_indices: Retrieved transaction indices, in ranking order.
    relevant_indices: Indices of transactions known to be relevant.
    """
    _validate_k(k)

    relevant = set(relevant_indices)
    top_results = retrieved_indices[:k]

    relevant_count = sum(
        1 for index in top_results if index in relevant
    )

    # Standard Precision@K uses K as the denominator.
    return relevant_count / k


def recall_at_k(
    retrieved_indices: Sequence[int],
    relevant_indices: Iterable[int],
    k: int = 3,
) -> float:
    """
    Calculate Recall@K.

    Returns 0.0 when no relevant items are defined.
    """
    _validate_k(k)

    relevant = set(relevant_indices)

    if not relevant:
        return 0.0

    top_results = retrieved_indices[:k]
    relevant_count = sum(
        1 for index in top_results if index in relevant
    )

    return relevant_count / len(relevant)


def reciprocal_rank(
    retrieved_indices: Sequence[int],
    relevant_indices: Iterable[int],
) -> float:
    """
    Calculate the reciprocal rank for one query.

    First relevant result at rank 1 -> 1.0
    First relevant result at rank 2 -> 0.5
    No relevant result -> 0.0
    """
    relevant = set(relevant_indices)

    for rank, index in enumerate(retrieved_indices, start=1):
        if index in relevant:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(
    query_results: Sequence[Sequence[int]],
    relevant_by_query: Sequence[Iterable[int]],
) -> float:
    """Calculate MRR across multiple queries."""
    if len(query_results) != len(relevant_by_query):
        raise ValueError(
            "Each query result must have a matching relevant set"
        )

    if not query_results:
        return 0.0

    reciprocal_ranks = [
        reciprocal_rank(results, relevant)
        for results, relevant in zip(
            query_results, relevant_by_query
        )
    ]

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def evaluate_dataset(
    evaluation_dataset: Sequence[Mapping[str, Any]],
    search_function: Callable[..., list[dict]] ,
    k: int = 3,
) -> dict[str, Any]:
    """
    Evaluate a search function against labeled queries.

    Each dataset item must contain:
        query: The search query.
        relevant_indices: Indices of expected relevant transactions.

    The search function must return dictionaries containing 'index'.
    """
    _validate_k(k)

    if not evaluation_dataset:
        raise ValueError("Evaluation dataset cannot be empty")

    query_results = []
    relevant_by_query = []
    per_query = []

    for item in evaluation_dataset:
        query = item["query"]
        relevant = set(item["relevant_indices"])

        results = search_function(query, top_k=k)
        retrieved_indices = [
            result["index"] for result in results
        ]

        precision = precision_at_k(
            retrieved_indices, relevant, k
        )
        recall = recall_at_k(
            retrieved_indices, relevant, k
        )
        rr = reciprocal_rank(
            retrieved_indices, relevant
        )

        query_results.append(retrieved_indices)
        relevant_by_query.append(relevant)

        per_query.append(
            {
                "query": query,
                "precision_at_k": precision,
                "recall_at_k": recall,
                "reciprocal_rank": rr,
                "retrieved_indices": retrieved_indices,
            }
        )

    return {
        "k": k,
        "precision_at_k": sum(
            row["precision_at_k"] for row in per_query
        ) / len(per_query),
        "recall_at_k": sum(
            row["recall_at_k"] for row in per_query
        ) / len(per_query),
        "mrr": mean_reciprocal_rank(
            query_results, relevant_by_query
        ),
        "per_query": per_query,
    }
