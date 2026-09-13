from src.semantic_search import search_transactions

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
