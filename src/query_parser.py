
import re

import numpy as np

from src.embeddings import create_embeddings


BRANCHES = ["Karachi", "Lahore", "Islamabad"]
TRANSACTION_TYPES = ["Deposit", "Withdrawal"]

INTENT_EXAMPLES = {
    "total_transaction_amount": [
        "What is the total transaction amount?",
        "Calculate the sum of all transactions",
        "What is the total amount of transactions?",
        "How much money was transacted overall?",
    ],
    "total_deposit": [
        "What is the total deposit amount?",
        "Calculate the sum of all deposits",
        "How much money was deposited?",
        "What is the total amount deposited?",
    ],
    "total_withdrawal": [
        "What is the total withdrawal amount?",
        "Calculate the sum of all withdrawals",
        "How much money was withdrawn?",
        "What is the total amount withdrawn?",
    ],
    "highest_transaction": [
        "What is the highest transaction?",
        "Find the largest transaction amount",
        "Which transaction has the maximum amount?",
    ],
    "lowest_transaction": [
        "What is the lowest transaction?",
        "Find the smallest transaction amount",
        "Which transaction has the minimum amount?",
    ],
    "most_active_branch": [
        "Which branch is the most active?",
        "Find the branch with the most transactions",
        "Which branch handles the most transactions?",
    ],
    "least_active_branch": [
        "Which branch is the least active?",
        "Find the branch with the fewest transactions",
        "Which branch handles the least transactions?",
    ],
    "transaction_lookup": [
        "Find transactions in Karachi",
        "Show withdrawal transactions",
        "Find transactions matching my query",
        "Show me a particular transaction",
    ],
}


def _build_embeddings(examples: dict) -> dict:
    """Build embeddings for the supplied examples."""

    embeddings = {}

    for label, phrases in examples.items():

        vectors = np.asarray(
            create_embeddings(phrases),
            dtype=np.float32,
        )

        if vectors.ndim != 2 or len(vectors) != len(phrases):
            raise ValueError(
                f"Invalid embeddings generated for {label}"
            )

        if not np.all(np.isfinite(vectors)):
            raise ValueError(
                f"Non-finite embeddings generated for {label}"
            )

        embeddings[label] = vectors

    return embeddings


INTENT_EMBEDDINGS = _build_embeddings(INTENT_EXAMPLES)

BRANCH_EMBEDDINGS = _build_embeddings({
    branch: [
        f"Transactions in {branch}",
        f"Financial activity at {branch} branch",
        f"Show me {branch} transactions",
    ]
    for branch in BRANCHES
})

TYPE_EMBEDDINGS = _build_embeddings({
    transaction_type: [
        f"Show {transaction_type.lower()} transactions",
        f"Transactions of type {transaction_type}",
        f"Amount of {transaction_type.lower()}",
    ]
    for transaction_type in TRANSACTION_TYPES
})


def _cosine_similarity(vector_a, vector_b) -> float:
    """Calculate cosine similarity between two vectors."""

    vector_a = np.asarray(vector_a, dtype=np.float32).reshape(-1)
    vector_b = np.asarray(vector_b, dtype=np.float32).reshape(-1)

    if vector_a.shape != vector_b.shape:
        raise ValueError("Embedding dimensions must match")

    if not np.all(np.isfinite(vector_a)) or not np.all(
        np.isfinite(vector_b)
    ):
        raise ValueError("Embeddings must contain finite values")

    denominator = (
        np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    )

    if denominator == 0:
        return 0.0

    return float(np.dot(vector_a, vector_b) / denominator)


def _find_best_match(
    query_embedding,
    embeddings: dict,
    minimum_score: float = 0.40,
):
    """Find the best semantic match among example embeddings."""

    if (
        isinstance(minimum_score, bool)
        or not isinstance(minimum_score, (int, float))
        or not np.isfinite(minimum_score)
        or not -1 <= minimum_score <= 1
    ):
        raise ValueError("minimum_score must be between -1 and 1")

    query_vector = np.asarray(
        query_embedding,
        dtype=np.float32,
    ).reshape(-1)

    if not np.all(np.isfinite(query_vector)):
        raise ValueError("Query embedding contains invalid values")

    if np.linalg.norm(query_vector) == 0:
        raise ValueError("Query embedding cannot be a zero vector")

    best_label = None
    best_score = -1.0

    for label, example_embeddings in embeddings.items():

        vectors = np.asarray(
            example_embeddings,
            dtype=np.float32,
        )

        if vectors.ndim != 2:
            raise ValueError("Example embeddings must be 2D")

        if vectors.shape[1] != query_vector.shape[0]:
            raise ValueError("Embedding dimensions must match")

        if not np.all(np.isfinite(vectors)):
            raise ValueError("Example embeddings contain invalid values")

        norms = np.linalg.norm(vectors, axis=1)

        similarities = np.divide(
            vectors @ query_vector,
            norms * np.linalg.norm(query_vector),
            out=np.zeros(len(vectors), dtype=np.float32),
            where=norms > 0,
        )

        if len(similarities) == 0:
            continue

        label_score = float(np.max(similarities))

        if label_score > best_score:
            best_label = label
            best_score = label_score

    if best_score < minimum_score:
        return None, best_score

    return best_label, best_score


def _find_branch(
    query: str,
    query_embedding=None,
) -> str | None:
    """Return a branch only when its name is explicitly mentioned."""

    for branch in BRANCHES:

        if re.search(
            rf"\b{re.escape(branch)}\b",
            query,
            re.IGNORECASE,
        ):
            return branch

    return None


def _find_transaction_type(
    query: str,
    query_embedding=None,
) -> str | None:
    """Detect transaction type from explicit query wording."""

    if re.search(
        r"\bwithdrawals?\b|\bwithdrawn\b",
        query,
        re.IGNORECASE,
    ):
        return "Withdrawal"

    if re.search(
        r"\bdeposits?\b|\bdeposited\b",
        query,
        re.IGNORECASE,
    ):
        return "Deposit"

    return None


def _extract_amount_filters(
    query: str,
) -> tuple[float | None, float | None]:
    """Extract minimum and maximum amount constraints."""

    normalized = query.lower().replace(",", "")

    number = r"\$?\s*(\d+(?:\.\d+)?)"
    currency = r"(?:rs\.?|pkr)?\s*"

    minimum_amount = None
    maximum_amount = None

    # Examples: above 5000, over 5000, at least 5000
    minimum_match = re.search(
        rf"\b(?:above|over|greater than|more than|at least|"
        rf"minimum of|min(?:imum)?)\s+{currency}{number}",
        normalized,
    )

    # Examples: below 10000, under 10000, at most 10000
    maximum_match = re.search(
        rf"\b(?:below|under|less than|at most|up to|"
        rf"maximum of|max(?:imum)?)\s+{currency}{number}",
        normalized,
    )

    # Examples: between 5000 and 10000, from 5000 to 10000
    range_match = re.search(
        rf"\b(?:between|from)\s+{currency}{number}"
        rf"\s+(?:and|to)\s+{currency}{number}",
        normalized,
    )

    if range_match:

        first = float(range_match.group(1))
        second = float(range_match.group(2))

        minimum_amount = min(first, second)
        maximum_amount = max(first, second)

    else:

        if minimum_match:
            minimum_amount = float(minimum_match.group(1))

        if maximum_match:
            maximum_amount = float(maximum_match.group(1))

    return minimum_amount, maximum_amount


def _detect_explicit_calculation(query: str) -> str | None:
    """Prioritize explicit calculation requests over semantic guesses."""

    query_lower = query.lower()

    has_total = bool(
        re.search(r"\b(total|sum)\b", query_lower)
    )

    if has_total and re.search(
        r"\b(withdrawals?|withdrawn)\b",
        query_lower,
    ):
        return "total_withdrawal"

    if has_total and re.search(
        r"\bdeposits?\b|\bdeposited\b",
        query_lower,
    ):
        return "total_deposit"

    if has_total and re.search(
        r"\btransactions?\b",
        query_lower,
    ):
        return "total_transaction_amount"

    if re.search(
        r"\b(how much|amount)\b",
        query_lower,
    ) and re.search(
        r"\b(withdrawals?|withdrawn)\b",
        query_lower,
    ):
        return "total_withdrawal"

    if re.search(
        r"\b(how much|amount)\b",
        query_lower,
    ) and re.search(
        r"\bdeposits?\b|\bdeposited\b",
        query_lower,
    ):
        return "total_deposit"

    return None


def parse_query(query: str) -> dict:
    """
    Parse a user query into intent, branch, transaction type,
    amount filters and a reusable query embedding.
    """

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query cannot be empty")

    query = query.strip()

    # Generate query embedding only once.
    query_embedding = np.asarray(
        create_embeddings([query]),
        dtype=np.float32,
    )

    if (
        query_embedding.ndim != 2
        or query_embedding.shape[0] != 1
    ):
        raise ValueError("Invalid query embedding shape")

    query_embedding = query_embedding[0]

    if not np.all(np.isfinite(query_embedding)):
        raise ValueError("Query embedding contains invalid values")

    if np.linalg.norm(query_embedding) == 0:
        raise ValueError("Query embedding cannot be a zero vector")

    # Semantic intent classification.
    intent, confidence = _find_best_match(
        query_embedding,
        INTENT_EMBEDDINGS,
    )

    if intent is None:
        intent = "transaction_lookup"

    # Explicit calculation wording takes priority.
    explicit_intent = _detect_explicit_calculation(query)

    if explicit_intent is not None:
        intent = explicit_intent

    # Extract explicitly mentioned entities.
    branch = _find_branch(query)

    transaction_type = _find_transaction_type(query)

    minimum_amount, maximum_amount = _extract_amount_filters(query)

    return {
        "intent": intent,
        "branch": branch,
        "transaction_type": transaction_type,
        "minimum_amount": minimum_amount,
        "maximum_amount": maximum_amount,
        "confidence": float(confidence),
        "query_embedding": query_embedding,
    }
