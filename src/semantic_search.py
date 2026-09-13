import numpy as np

from src.embeddings import create_embeddings
from src.embedding_store import load_embeddings
from src.logger import logger


def search_transactions(
    query: str,
    top_k: int = 3,
    branch: str | None = None,
    transaction_type: str | None = None,
    minimum_score:float=0.40
):
    """
    Find the most relevant transactions for a user query.

    Optional filters:
        branch: Filter by branch, e.g. "Karachi"
        transaction_type: Filter by type, e.g. "Withdrawal"
    """

    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    texts, embeddings = load_embeddings()

    query_embedding = create_embeddings([query])[0]

    query_norm = np.linalg.norm(query_embedding)
    embedding_norms = np.linalg.norm(embeddings, axis=1)

    similarities = (
        embeddings @ query_embedding
    ) / (embedding_norms * query_norm)

    candidates = []

    for index, text in enumerate(texts):

        if branch and f"Branch: {branch}" not in text:
            continue

        if transaction_type and f"Type: {transaction_type}" not in text:
            continue

        candidates.append(
            {
                "index": index,
                "text": text,
                "score": float(similarities[index])
            }
        )

    candidates.sort(
        key=lambda result: result["score"],
        reverse=True
    )

    results = [result
                         for result in candidates
                         if result["score"]>=minimum_score][:top_k]

    logger.info(
        f"Semantic search completed for query: '{query}'"
    )

    return results