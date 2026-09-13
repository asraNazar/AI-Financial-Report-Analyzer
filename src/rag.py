from openai import OpenAI, OpenAIError

from config import OPENAI_API_KEY

from src.semantic_search import search_transactions

from src.logger import logger
from src.analyzer import analyze_transactions


client = OpenAI(api_key=OPENAI_API_KEY)


def extract_filters(query: str):
    """
    Extract branch and transaction type from a user query.
    """

    query_lower = query.lower()

    branch = None
    transaction_type = None

    branches = [
        "Karachi",
        "Lahore",
        "Islamabad"
    ]

    transaction_types = [
        "Deposit",
        "Withdrawal"
    ]

    for item in branches:
        if item.lower() in query_lower:
            branch = item
            break

    for item in transaction_types:
        if item.lower() in query_lower:
            transaction_type = item
            break

    return branch, transaction_type

def classify_question(query: str):
    """
    Classify a financial question into a basic question type.
    """

    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    query_lower = query.lower()

    calculation_keywords = [
        "total",
        "sum",
        "average",
        "difference",
        "highest",
        "lowest",
        "maximum",
        "minimum",
        "how much",
        "amount"
    ]

    for keyword in calculation_keywords:
        if keyword in query_lower:
            return "calculation"

    return "transaction"

def calculate_financial_answer(df, query: str):
    """
    Calculate financial metrics based on the user's question.
    """

    results = analyze_transactions(df)

    query_lower = query.lower()

    if "withdrawal" in query_lower and "total" in query_lower:
        return (
            f"Total withdrawal amount is "
            f"{results['total_withdrawals']:,.0f}."
        )

    if "deposit" in query_lower and "total" in query_lower:
        return (
            f"Total deposit amount is "
            f"{results['total_deposits']:,.0f}."
        )

    if "total transaction amount" in query_lower:
        return (
            f"Total transaction amount is "
            f"{results['total_amount']:,.0f}."
        )

    if "highest transaction" in query_lower:
        return (
            f"The highest transaction amount is "
            f"{results['highest_transactions']:,.0f}."
        )

    if "lowest transaction" in query_lower:
        return (
            f"The lowest transaction amount is "
            f"{results['lowest_transactions']:,.0f}."
        )

    if "most active branch" in query_lower:
        return (
            f"The most active branch is "
            f"{results['most_active_branch']}."
        )

    if "least active branch" in query_lower:
        return (
            f"The least active branch is "
            f"{results['least_active_branch']}."
        )

    return "I could not determine the requested calculation."

def retrieve_context(
    query: str,
    top_k: int = 3,
    branch: str | None = None,
    transaction_type: str | None = None
):
    """
    Retrieve relevant transactions for a user query.

    Automatically detects branch and transaction type
    when they are not explicitly provided.
    """

    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    detected_branch, detected_type = extract_filters(query)

    if branch is None:
        branch = detected_branch

    if transaction_type is None:
        transaction_type = detected_type

    results = search_transactions(
        query,
        top_k=top_k,
        branch=branch,
        transaction_type=transaction_type
    )

    if not results:
        logger.info("No relevant transactions found")
        return ""

    context = "\n".join(
        f"- {result['text']} "
        f"(similarity: {result['score']:.2f})"
        for result in results
    )

    logger.info(
        f"Retrieved {len(results)} relevant transactions"
    )

    return context


def ask_rag(query: str,df=None, top_k: int = 3):
    """
    Answer a user question using retrieved transaction context.
    """

    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    question_type = classify_question(query)

    if question_type == "calculation":

        if df is None:
            return (
                "Calculation requires transaction data."
            )

        return calculate_financial_answer(
            df,
            query
        )
    context = retrieve_context(
        query,
        top_k=top_k
    )

    if not context:
        return "I could not find relevant transactions."

    prompt = f"""
You are a financial data assistant.

Answer the user's question using ONLY the transaction data
provided in the context below.

Do not invent transactions or financial information.

If the context does not contain enough information to answer
the question, clearly say that the available data is insufficient.

Transaction Context:

{context}

User Question:

{query}

Provide a clear and concise answer.
"""

    try:
        response = client.responses.create(
            model="gpt-5.5",
            input=prompt
        )

        answer = response.output_text

        logger.info("RAG answer generated successfully")

        return answer

    except OpenAIError as error:
        logger.error(f"OpenAI API error: {error}")

        clean_data = get_clean_transaction_data(
            query,
            top_k=top_k
        )

        return (
            "AI-generated answer is currently unavailable.\n\n"
            f"{clean_data}"
        )


def test_rag(query: str, top_k: int = 3):
    """
    Test the RAG retrieval pipeline without calling OpenAI API.
    """

    context = retrieve_context(
        query,
        top_k=top_k
    )

    if not context:
        return "I could not find relevant transactions."

    answer = f"""
Based on the retrieved transaction data:

{context}
"""

    return answer


def get_clean_transaction_data(
    query: str,
    top_k: int = 3
):
    """
    Retrieve transactions and return clean user-facing data.
    """

    branch, transaction_type = extract_filters(query)

    results = search_transactions(
        query,
        top_k=top_k,
        branch=branch,
        transaction_type=transaction_type
    )

    if not results:
        return "I could not find relevant transactions."

    formatted_transactions = []

    for result in results:

        text = result["text"]

        parts = [
            part.strip()
            for part in text.split("|")
        ]

        transaction = {}

        for part in parts:
            if ":" in part:
                key, value = part.split(":", 1)
                transaction[key.strip()] = value.strip()

        amount = transaction.get("Amount")

        if amount:
            try:
                amount = f"{float(amount):,.0f}"
            except ValueError:
                pass

        formatted_transaction = (
            f"Date: {transaction.get('Date', 'N/A')}\n"
            f"Branch: {transaction.get('Branch', 'N/A')}\n"
            f"Type: {transaction.get('Type', 'N/A')}\n"
            f"Amount: {amount if amount else 'N/A'}"
        )

        formatted_transactions.append(formatted_transaction)

    formatted_transactions = "\n\n".join(
        formatted_transactions
    )

    return (
        "Relevant transaction data:\n\n"
        f"{formatted_transactions}"
    )
