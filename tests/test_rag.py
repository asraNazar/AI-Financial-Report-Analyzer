import pytest

from src.rag import (
    retrieve_context,
    ask_rag,
    extract_filters,classify_question,calculate_financial_answer
)
from src.semantic_search import search_transactions


def test_retrieve_karachi_withdrawal():
    context = retrieve_context(
        "withdrawal",
        branch="Karachi",
        transaction_type="Withdrawal"
    )

    assert context != ""
    assert "Branch: Karachi" in context
    assert "Type: Withdrawal" in context
    assert "Amount: 90000" in context


def test_retrieve_karachi_deposit():
    context = retrieve_context(
        "deposit",
        branch="Karachi",
        transaction_type="Deposit"
    )

    assert context != ""
    assert "Branch: Karachi" in context
    assert "Type: Deposit" in context
    assert "Amount: 50000" in context


def test_retrieve_no_matching_transaction():
    context = retrieve_context(
        "deposit",
        branch="Lahore",
        transaction_type="Deposit"
    )

    assert context == ""


def test_retrieve_empty_query():
    with pytest.raises(ValueError):
        retrieve_context("")

def test_ask_rag_with_mocked_openai(monkeypatch):
    class FakeResponse:
        output_text = "Karachi had a withdrawal of 90000."

    def fake_create(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr(
        "src.rag.client.responses.create",
        fake_create
    )

    answer = ask_rag(
        "What withdrawal happened in Karachi?",
        top_k=1
    )

    assert answer == "Karachi had a withdrawal of 90000."

def test_extract_karachi_withdrawal():
    branch, transaction_type = extract_filters(
        "What withdrawal happened in Karachi?"
    )

    assert branch == "Karachi"
    assert transaction_type == "Withdrawal"


def test_extract_lahore_deposit():
    branch, transaction_type = extract_filters(
        "Show me deposits in Lahore"
    )

    assert branch == "Lahore"
    assert transaction_type == "Deposit"


def test_extract_no_filters():
    branch, transaction_type = extract_filters(
        "Show me financial transactions"
    )

    assert branch is None
    assert transaction_type is None


def test_ask_rag_handles_openai_error(monkeypatch):
    def fake_create(*args, **kwargs):
        from openai import OpenAIError
        raise OpenAIError("API unavailable")

    monkeypatch.setattr(
        "src.rag.client.responses.create",
        fake_create
    )

    answer = ask_rag(
        "What withdrawal happened in Karachi?",
        top_k=1
    )

    assert "AI-generated answer is currently unavailable" in answer
    assert "Branch: Karachi" in answer
    assert "Type: Withdrawal" in answer
    assert "Amount: 90,000" in answer

def test_get_clean_transaction_data():
    from src.rag import get_clean_transaction_data

    result = get_clean_transaction_data(
        "withdrawal in Lahore"
    )

    assert "Date: 2026-01-02" in result
    assert "Branch: Lahore" in result
    assert "Type: Withdrawal" in result
    assert "Amount: 20,000" in result
    assert "similarity" not in result

def test_classify_transaction_question():
    question_type = classify_question(
        "What withdrawal happened in Karachi?"
    )

    assert question_type == "transaction"


def test_classify_calculation_question():
    question_type = classify_question(
        "What is the total withdrawal amount?"
    )

    assert question_type == "calculation"


def test_classify_empty_question():
    with pytest.raises(ValueError):
        classify_question("")

def test_calculate_total_deposit():
    from src.data_loader import load_excel
    from src.validator import validate_dataframe
    from src.cleaner import clean_dataFrame
    from config import DATA_DIR

    df = load_excel(DATA_DIR / "transactions.xlsx")
    validate_dataframe(df)
    df = clean_dataFrame(df)

    result = calculate_financial_answer(
        df,
        "What is the total deposit amount?"
    )

    assert result == "Total deposit amount is 65,000."


def test_calculate_total_withdrawal():
    from src.data_loader import load_excel
    from src.cleaner import clean_dataFrame
    from config import DATA_DIR
    from src.validator import validate_dataframe

    df = load_excel(DATA_DIR / "transactions.xlsx")
    validate_dataframe(df)
    df = clean_dataFrame(df)

    result = calculate_financial_answer(
        df,
        "What is the total withdrawal amount?"
    )

    assert result == "Total withdrawal amount is 110,000."


def test_calculate_total_transaction_amount():
    from src.data_loader import load_excel
    from src.cleaner import clean_dataFrame
    from config import DATA_DIR
    from src.validator import validate_dataframe

    df = load_excel(DATA_DIR / "transactions.xlsx")
    validate_dataframe(df)
    df = clean_dataFrame(df)

    result = calculate_financial_answer(
        df,
        "What is the total transaction amount?"
    )

    assert result == "Total transaction amount is 175,000."