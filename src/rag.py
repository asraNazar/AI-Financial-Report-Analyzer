from openai import OpenAI, OpenAIError

from config import OPENAI_API_KEY

from src.semantic_search import search_transactions
from src.logger import logger
from src.analyzer import analyze_transactions
from src.query_parser import parse_query


client = OpenAI(api_key=OPENAI_API_KEY)


def extract_filters(query: str):
    """
    Extract branch and transaction type from a user query.

    Uses the central query parser instead of maintaining
    separate filter-detection logic.
    """
    parsed_query = parse_query(query)

    return (
        parsed_query["branch"],
        parsed_query["transaction_type"],
    )


def classify_question(query: str):
    """
    Classify a user question using the central query parser.
    """
    parsed_query = parse_query(query)

    if parsed_query["intent"] == "transaction_lookup":
        return "transaction"

    return "calculation"


def calculate_financial_answer(
    df,
    query: str,
    parsed_query: dict | None = None,
):
    """
    Calculate financial metrics based on the user's question.

    Supports overall and branch-specific calculations.
    """
    if parsed_query is None:
        parsed_query = parse_query(query)

    intent = parsed_query["intent"]
    branch = parsed_query["branch"]
    transaction_type = parsed_query["transaction_type"]

    filtered_df = df

    # Filter by branch
    if branch:
        filtered_df = filtered_df[
            filtered_df["Branch"].str.lower() == branch.lower()
        ]

    # Filter by transaction type
    if transaction_type:
        filtered_df = filtered_df[
            filtered_df["Type"].str.lower()
            == transaction_type.lower()
        ]

    location = f" in {branch}" if branch else ""

    # Total calculations
    total_intents = {
        "total_withdrawal": "Total withdrawal amount",
        "total_deposit": "Total deposit amount",
        "total_transaction_amount": "Total transaction amount",
    }

    if intent in total_intents:
        amount = filtered_df["Amount"].sum()

        return (
            f"{total_intents[intent]}{location} is "
            f"{amount:,.0f}."
        )

    # Highest / lowest transaction
    if intent in {
        "highest_transaction",
        "lowest_transaction",
    }:
        if filtered_df.empty:
            return "No matching transactions found."

        if intent == "highest_transaction":
            amount = filtered_df["Amount"].max()
            label = "highest"

        else:
            amount = filtered_df["Amount"].min()
            label = "lowest"

        return (
            f"The {label} transaction amount{location} is "
            f"{amount:,.0f}."
        )

    # Branch activity calculations
    if intent in {
        "most_active_branch",
        "least_active_branch",
    }:
        results = analyze_transactions(df)

        if intent == "most_active_branch":
            return (
                f"The most active branch is "
                f"{results['most_active_branch']}."
            )

        return (
            f"The least active branch is "
            f"{results['least_active_branch']}."
        )

    return "I could not determine the requested calculation."


def retrieve_context(
    query: str,
    top_k: int = 3,
    branch: str | None = None,
    transaction_type: str | None = None,
    parsed_query: dict | None = None,
):
    """
    Retrieve relevant transactions for a user query.

    If parsed_query is provided, its branch, transaction type,
    and query embedding are reused.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    if parsed_query is None:
        parsed_query = parse_query(query)

    # Explicit arguments override parsed values.
    selected_branch = (
        branch
        if branch is not None
        else parsed_query["branch"]
    )

    selected_type = (
        transaction_type
        if transaction_type is not None
        else parsed_query["transaction_type"]
    )

    results = search_transactions(
        query,
        top_k=top_k,
        branch=selected_branch,
        transaction_type=selected_type,
        query_embedding=parsed_query["query_embedding"],
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


def ask_rag(
    query: str,
    df=None,
    top_k: int = 3,
):
    """
    Answer a user question using retrieved transaction context.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    # Parse the query only once.
    parsed_query = parse_query(query)

    if parsed_query["intent"] != "transaction_lookup":

        if df is None:
            return "Calculation requires transaction data."

        return calculate_financial_answer(
            df,
            query,
            parsed_query=parsed_query,
        )

    # Reuse parsed query and query embedding.
    context = retrieve_context(
        query,
        top_k=top_k,
        parsed_query=parsed_query,
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
            input=prompt,
        )

        answer = response.output_text

        logger.info(
            "RAG answer generated successfully"
        )

        return answer

    except OpenAIError as error:
        logger.error(
            f"OpenAI API error: {error}"
        )

        clean_data = get_clean_transaction_data(
            query,
            top_k=top_k,
            parsed_query=parsed_query,
        )

        return (
            "AI-generated answer is currently unavailable.\n\n"
            f"{clean_data}"
        )


def test_rag(
    query: str,
    top_k: int = 3,
):
    """
    Test the RAG retrieval pipeline without calling OpenAI API.
    """
    parsed_query = parse_query(query)

    context = retrieve_context(
        query,
        top_k=top_k,
        parsed_query=parsed_query,
    )

    if not context:
        return "I could not find relevant transactions."

    return (
        "Based on the retrieved transaction data:\n\n"
        f"{context}"
    )


def get_clean_transaction_data(
    query: str,
    top_k: int = 3,
    parsed_query: dict | None = None,
):
    """
    Retrieve transactions and return clean user-facing data.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    if parsed_query is None:
        parsed_query = parse_query(query)

    results = search_transactions(
        query,
        top_k=top_k,
        branch=parsed_query["branch"],
        transaction_type=parsed_query["transaction_type"],
        query_embedding=parsed_query["query_embedding"],
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

        formatted_transactions.append(
            formatted_transaction
        )

    formatted_transactions = "\n\n".join(
        formatted_transactions
    )

    return (
        "Relevant transaction data:\n\n"
        f"{formatted_transactions}"
    )