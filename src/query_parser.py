import numpy as np

from src.embeddings import create_embeddings
from src.logger import logger


INTENT_EXAMPLES = {
    "total_transaction_amount": [
        "What is the total transaction amount?",
        "How much money was involved in all transactions?",
        "What is the total amount of transactions?",
        "How much was the total transaction value?"
    ],

    "total_deposit": [
        "What is the total deposit amount?",
        "How much money was deposited?",
        "What is the total amount deposited?",
        "How much was deposited?"
    ],

    "total_withdrawal": [
        "What is the total withdrawal amount?",
        "How much money was withdrawn?",
        "What is the total amount withdrawn?",
        "How much was withdrawn?"
    ],

    "highest_transaction": [
        "What is the highest transaction?",
        "Which transaction has the highest amount?",
        "What was the largest transaction?",
        "What is the maximum transaction amount?"
    ],

    "lowest_transaction": [
        "What is the lowest transaction?",
        "Which transaction has the lowest amount?",
        "What was the smallest transaction?",
        "What is the minimum transaction amount?"
    ],
    "transaction_lookup": [
    "What transactions happened?",
    "Show me the transactions",
    "Which transaction occurred?",
    "What happened in the transactions?",
    "What withdrawal happened?",
    "What deposit happened?",
    "Show me a withdrawal transaction",
    "Show me a deposit transaction"
]
}


BRANCHES = [
    "Karachi",
    "Lahore",
    "Islamabad"
]

BRANCH_EXAMPLES = {
    "Karachi": [
        "transactions in Karachi",
        "money movement at Karachi branch",
        "Karachi branch transactions",
        "financial activity in Karachi"
    ],

    "Lahore": [
        "transactions in Lahore",
        "money movement at Lahore branch",
        "Lahore branch transactions",
        "financial activity in Lahore"
    ],

    "Islamabad": [
        "transactions in Islamabad",
        "money movement at Islamabad branch",
        "Islamabad branch transactions",
        "financial activity in Islamabad"
    ]
}

TRANSACTION_TYPES = [
    "Deposit",
    "Withdrawal"
]

TRANSACTION_TYPE_EXAMPLES = {
    "Deposit": [
        "money was deposited",
        "cash was added",
        "money came into the account",
        "funds were added"
    ],
    "Withdrawal": [
        "money was withdrawn",
        "money was taken out",
        "cash was taken from the account",
        "funds were removed"
    ]
}

def _build_intent_embeddings():
    """
    Create embeddings for all predefined intent examples.
    """

    labels = []
    examples = []

    for intent, intent_examples in INTENT_EXAMPLES.items():
        for example in intent_examples:
            labels.append(intent)
            examples.append(example)

    embeddings = create_embeddings(examples)

    return labels, embeddings


INTENT_LABELS, INTENT_EMBEDDINGS = _build_intent_embeddings()

def _build_transaction_type_embeddings():
    """
    Create embeddings for transaction type examples.
    """

    labels = []
    examples = []

    for transaction_type, type_examples in TRANSACTION_TYPE_EXAMPLES.items():
        for example in type_examples:
            labels.append(transaction_type)
            examples.append(example)

    embeddings = create_embeddings(examples)

    return labels, embeddings
TRANSACTION_TYPE_LABELS, TRANSACTION_TYPE_EMBEDDINGS = (
    _build_transaction_type_embeddings()
)

def _build_branch_embeddings():
    """
    Create embeddings for branch examples.
    """
    labels=[]
    examples=[]

    for branch,branch_examples in BRANCH_EXAMPLES.items():
        for example in branch_examples:
            labels.append(branch)
            examples.append(example)
    embeddings = create_embeddings(examples)
    return labels,embeddings

BRANCH_LABELS,BRANCH_EMBEDDINGS = _build_branch_embeddings()

def _find_branch(query: str):
    """
    Detect branch using semantic similarity.
    Return None when no branch is sufficiently relevant.
    """

    query_embedding = create_embeddings([query])[0]

    query_norm = np.linalg.norm(query_embedding)

    branch_norms = np.linalg.norm(
        BRANCH_EMBEDDINGS,
        axis=1
    )

    similarities = (
        BRANCH_EMBEDDINGS @ query_embedding
    ) / (
        branch_norms * query_norm
    )

    best_index = int(np.argmax(similarities))
    best_score = float(similarities[best_index])

    logger.info(
        f"Best branch match: "
        f"{BRANCH_LABELS[best_index]} "
        f"(score={best_score:.4f})"
    )

    if best_score < 0.65:
        return None

    return BRANCH_LABELS[best_index]

def _find_transaction_type(query: str):
    """
    Detect transaction type using semantic similarity.
    """

    query_embedding = create_embeddings([query])[0]

    query_norm = np.linalg.norm(query_embedding)
    type_norms = np.linalg.norm(
        TRANSACTION_TYPE_EMBEDDINGS,
        axis=1
    )

    similarities = (
        TRANSACTION_TYPE_EMBEDDINGS @ query_embedding
    ) / (
        type_norms * query_norm
    )

    best_index = int(np.argmax(similarities))

    return TRANSACTION_TYPE_LABELS[best_index]

def parse_query(query: str):
    """
    Understand a financial query using semantic similarity.
    """

    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    query_embedding = create_embeddings([query])[0]

    query_norm = np.linalg.norm(query_embedding)

    intent_norms = np.linalg.norm(
        INTENT_EMBEDDINGS,
        axis=1
    )

    similarities = (
        INTENT_EMBEDDINGS @ query_embedding
    ) / (
        intent_norms * query_norm
    )

    best_index = int(np.argmax(similarities))

    intent = INTENT_LABELS[best_index]
    confidence = float(similarities[best_index])

    branch = _find_branch(query)

    if intent in {
        "total_deposit",
        "total_withdrawal"
    }:
        transaction_type = _find_transaction_type(query)
    else:
        transaction_type = None

    result = {
        "intent": intent,
        "branch": branch,
        "transaction_type": transaction_type,
        "confidence": confidence
    }

    logger.info(
        f"Query parsed: {result}"
    )

    return result