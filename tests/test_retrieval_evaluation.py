
import pytest

from src.retrieval_evaluation import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    mean_reciprocal_rank,
    evaluate_dataset,
)


def test_precision_at_k():
    assert precision_at_k([2, 4, 1], {2, 1}, k=3) == pytest.approx(
        2 / 3
    )


def test_recall_at_k():
    assert recall_at_k([2, 4, 1], {2, 1, 5}, k=3) == pytest.approx(
        2 / 3
    )


def test_reciprocal_rank_first_result():
    assert reciprocal_rank([2, 4, 1], {2}) == 1.0


def test_reciprocal_rank_second_result():
    assert reciprocal_rank([4, 2, 1], {2}) == 0.5


def test_reciprocal_rank_no_relevant_result():
    assert reciprocal_rank([4, 3, 1], {2}) == 0.0


def test_mean_reciprocal_rank():
    results = [[2, 4, 1], [4, 2, 3]]
    relevant = [{2}, {2}]

    assert mean_reciprocal_rank(results, relevant) == pytest.approx(
        0.75
    )


def test_empty_relevant_set():
    assert recall_at_k([1, 2], set(), k=2) == 0.0


def test_invalid_k():
    with pytest.raises(ValueError):
        precision_at_k([1, 2], {1}, k=0)


def test_mismatched_mrr_inputs():
    with pytest.raises(ValueError):
        mean_reciprocal_rank([[1, 2]], [])


def test_evaluate_dataset():
    dataset = [
        {"query": "query one", "relevant_indices": {2}},
        {"query": "query two", "relevant_indices": {3}},
    ]

    def fake_search(query, top_k=3):
        if query == "query one":
            return [{"index": 2}, {"index": 4}]
        return [{"index": 1}, {"index": 3}]

    metrics = evaluate_dataset(dataset, fake_search, k=2)

    assert metrics["precision_at_k"] == pytest.approx(0.5)
    assert metrics["recall_at_k"] == pytest.approx(1.0)
    assert metrics["mrr"] == pytest.approx(0.75)
    assert len(metrics["per_query"]) == 2
