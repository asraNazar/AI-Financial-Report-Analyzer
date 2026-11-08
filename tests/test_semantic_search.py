from src.semantic_search import search_transactions
import numpy as np
import pytest

def test_search_karachi_withdrawal():
    results = search_transactions(
        "withdrawal",branch="Karachi",
        transaction_type="Withdrawal"
    )

    assert len(results)==1
    assert "Branch: Karachi" in results[0]["text"]
    assert "Type: Withdrawal" in results[0]["text"]
    assert "Amount: 90000" in results[0]["text"]

def test_search_karachi_deposit():
    results = search_transactions(
        "deposit",branch="Karachi",transaction_type="Deposit"
    )
    assert len(results) == 1
    assert "Branch: Karachi" in results[0]["text"]
    assert "Type: Deposit" in results[0]["text"]
    assert "Amount: 50000" in results[0]["text"]


def test_search_lahore_withdrawal():
    results = search_transactions(
        "withdrawal",
        branch="Lahore",
        transaction_type="Withdrawal"
    )

    assert len(results) == 1
    assert "Branch: Lahore" in results[0]["text"]
    assert "Type: Withdrawal" in results[0]["text"]
    assert "Amount: 20000" in results[0]["text"]


def test_search_with_amount_range():
    results = search_transactions(
        "transactions",
        minimum_amount=40000,
        maximum_amount=60000
    )

    assert all(
        40000 <= float(
            result["text"].split("Amount: ")[1].split("|")[0]
        ) <= 60000
        for result in results
    )


def test_search_invalid_query():
    with pytest.raises(ValueError):
        search_transactions("")


def test_search_invalid_top_k():
    with pytest.raises(ValueError):
        search_transactions("withdrawal", top_k=0)


def test_search_invalid_amount_range():
    with pytest.raises(ValueError):
        search_transactions(
            "transactions",
            minimum_amount=90000,
            maximum_amount=10000
        )


def test_search_invalid_query_embedding():
    with pytest.raises(ValueError):
        search_transactions(
            "withdrawal",
            query_embedding=np.zeros(384)
        )

def test_hybrid_search_returns_all_scores():
    results = search_transactions(
        "Karachi withdrawal",
        branch="Karachi",
        transaction_type="Withdrawal",
    )

    assert len(results) == 1

    result = results[0]

    assert "score" in result
    assert "semantic_score" in result
    assert "keyword_score" in result

    assert 0 <= result["score"] <= 1
    assert result["keyword_score"] > 0


def test_keyword_match_can_retrieve_below_semantic_threshold(
    monkeypatch,
):
    import src.semantic_search as semantic_search

    texts = [
        "Branch: Karachi | Type: Withdrawal | Amount: 90000",
        "Branch: Lahore | Type: Deposit | Amount: 50000",
    ]

    embeddings = np.array([
        [0.0, 1.0],
        [1.0, 0.0],
    ])

    monkeypatch.setattr(
        semantic_search,
        "load_embeddings",
        lambda: (texts, embeddings),
    )

    results = search_transactions(
        "withdrawal",
        query_embedding=np.array([1.0, 0.0]),
        minimum_score=0.99,
    )

    withdrawal_results = [
        result for result in results
        if "Withdrawal" in result["text"]
    ]

    assert len(withdrawal_results) == 1
    assert withdrawal_results[0]["semantic_score"] < 0.99
    assert withdrawal_results[0]["keyword_score"] > 0