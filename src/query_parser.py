import numpy as np

from src.embeddings import create_embeddings
from src.logger import logger


INTENT_EXAMPLES = {
    "total_transaction_amount": [
        "What is the total transaction amount?",
        "How much money was involved in all transactions?",
        "What is the total amount of transactions?",
        "How much was the total transaction value?",
    ],
    "total_deposit": [
        "What is the total deposit amount?",
        "How much money was deposited?",
        "What is the total amount deposited?",
        "How much was deposited?",
    ],
    "total_withdrawal": [
        "What is the total withdrawal amount?",
        "How much money was withdrawn?",
        "What is the total amount withdrawn?",
        "How much was withdrawn?",
    ],
    "highest_transaction": [
        "What is the highest transaction?",
        "Which transaction has the highest amount?",
        "What was the largest transaction?",
        "What is the maximum transaction amount?",
    ],
    "lowest_transaction": [
        "What is the lowest transaction?",
        "Which transaction has the lowest amount?",
        "What was the smallest transaction?",
        "What is the minimum transaction amount?",
    ],
    "transaction_lookup": [
        "What transactions happened?",
        "Show me the transactions",
        "Which transaction occurred?",
        "What happened in the transactions?",
        "What withdrawal happened?",
        "What deposit happened?",
        "Show me a withdrawal transaction",
        "Show me a deposit transaction",
    ],
}


BRANCH_EXAMPLES = {
    "Karachi": [
        "transactions in Karachi",
        "money movement at Karachi branch",
        "Karachi branch transactions",
        "financial activity in Karachi",
    ],
    "Lahore": [
        "transactions in Lahore",
        "money movement at Lahore branch",
        "Lahore branch transactions",
        "financial activity in Lahore",
    ],
    "Islamabad": [
        "transactions in Islamabad",
        "money movement at Islamabad branch",
        "Islamabad branch transactions",
        "financial activity in Islamabad",
    ],
}


TRANSACTION_TYPE_EXAMPLES = {
    "Deposit": [
        "money was deposited",
        "cash was added",
        "money came into the account",
        "funds were added",
    ],
    "Withdrawal": [
        "money was withdrawn",
        "money was taken out",
        "cash was taken from the account",
        "funds were removed",
    ],
}


def _build_embeddings(example_groups):
    """
    Build labels and embeddings from grouped examples.
    """
    labels = []
    examples = []

    for label, label_examples in example_groups.items():
        for example in label_examples:
            labels.append(label)
            examples.append(example)

    embeddings = create_embeddings(examples)

    return labels, embeddings


INTENT_LABELS, INTENT_EMBEDDINGS = _build_embeddings(
    INTENT_EXAMPLES
)

BRANCH_LABELS, BRANCH_EMBEDDINGS = _build_embeddings(
    BRANCH_EXAMPLES
)

TRANSACTION_TYPE_LABELS, TRANSACTION_TYPE_EMBEDDINGS = (
    _build_embeddings(TRANSACTION_TYPE_EXAMPLES)
)


def _find_best_match(
    query_embedding,
    labels,
    embeddings,
    minimum_score=None,
):
    """
    Find the most similar label using a pre-computed
    query embedding.
    """
    query_norm = np.linalg.norm(query_embedding)

    embedding_norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    similarities = (
        embeddings @ query_embedding
    ) / (
        embedding_norms * query_norm
    )

    best_index = int(np.argmax(similarities))
    best_score = float(similarities[best_index])

    if (
        minimum_score is not None
        and best_score < minimum_score
    ):
        return None, best_score

    return labels[best_index], best_score


def _find_branch(query_embedding):
    """
    Detect branch using semantic similarity.
    """
    branch, score = _find_best_match(
        query_embedding,
        BRANCH_LABELS,
        BRANCH_EMBEDDINGS,
        minimum_score=0.65,
    )

    if branch:
        logger.info(
            f"Best branch match: {branch} "
            f"(score={score:.4f})"
        )

    return branch


def _find_transaction_type(query_embedding):
    """
    Detect transaction type using semantic similarity.
    """
    transaction_type, score = _find_best_match(
        query_embedding,
        TRANSACTION_TYPE_LABELS,
        TRANSACTION_TYPE_EMBEDDINGS,
        minimum_score=0.65,
    )

    if transaction_type:
        logger.info(
            f"Best transaction type match: {transaction_type} "
            f"(score={score:.4f})"
        )

    return transaction_type


def parse_query(query: str):
    """
    Understand a financial query using semantic similarity.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    # Create the query embedding only once.
    query_embedding = create_embeddings([query])[0]

    intent, confidence = _find_best_match(
        query_embedding,
        INTENT_LABELS,
        INTENT_EMBEDDINGS,
    )

    
    branch = _find_branch(query_embedding)

    query_lower = query.lower()

    # Detect transaction type directly from the query.
    if "deposit" in query_lower:
        transaction_type = "Deposit"

    elif "withdrawal" in query_lower or "withdraw" in query_lower:
        transaction_type = "Withdrawal"

    elif intent == "total_deposit":
        transaction_type = "Deposit"

    elif intent == "total_withdrawal":
        transaction_type = "Withdrawal"

    else:
        transaction_type = None


    result = {
        "intent": intent,
        "branch": branch,
        "transaction_type": transaction_type,
        "confidence": confidence,
        "query_embedding": query_embedding,
    }

    logger.info(f"Query parsed: {result}")

    return result