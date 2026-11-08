import re
import numpy as np
from collections import Counter
from math import log

from src.embeddings import create_embeddings
from src.embedding_store import load_embeddings
from src.logger import logger


SEMANTIC_WEIGHT = 0.70
KEYWORD_WEIGHT = 0.30

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were",
    "in", "at", "on", "of", "for", "to", "and",
    "show", "find", "me", "my", "what", "which",
    "all", "please", "get", "list",
    "transaction", "transactions",
    "date", "branch", "type", "amount",
}


def _validate_amount(value, name):
    if value is None:
        return

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a valid number or None")

    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _validate_filter(value, name):
    if value is not None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{name} must be a non-empty string or None"
            )


def _extract_field(text, field):
    pattern = rf"(?:^|\|)\s*{re.escape(field)}:\s*([^|]*)"
    match = re.search(pattern, text, flags=re.IGNORECASE)

    return match.group(1).strip() if match else None


def _extract_amount(text):
    amount = _extract_field(text, "Amount")

    if amount is None:
        return None

    try:
        value = float(amount.replace(",", "").strip())
        return value if np.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def _validate_query_embedding(query_embedding, embedding_dimension):
    query_embedding = np.asarray(query_embedding, dtype=float)

    if query_embedding.ndim != 1:
        raise ValueError(
            "Query embedding must be a one-dimensional array"
        )

    if query_embedding.shape[0] != embedding_dimension:
        raise ValueError(
            "Query embedding dimension does not match stored embeddings"
        )

    if not np.all(np.isfinite(query_embedding)):
        raise ValueError("Query embedding contains invalid values")

    query_norm = np.linalg.norm(query_embedding)

    if query_norm == 0:
        raise ValueError("Query embedding cannot be a zero vector")

    return query_embedding, query_norm


def _tokenize(text):
    tokens = re.findall(r"[a-zA-Z]+|\d+(?:\.\d+)?", text.casefold())

    return [
        token for token in tokens
        if token not in STOP_WORDS
    ]


def _calculate_bm25(query, documents, k1=1.5, b=0.75):
    """
    Calculate BM25-style keyword relevance scores.

    Returns one raw keyword score per document.
    """

    query_tokens = _tokenize(query)

    if not query_tokens or not documents:
        return np.zeros(len(documents), dtype=float)

    document_tokens = [_tokenize(doc) for doc in documents]
    document_lengths = np.array(
        [len(tokens) for tokens in document_tokens],
        dtype=float,
    )

    average_length = (
        float(np.mean(document_lengths))
        if len(document_lengths) and np.mean(document_lengths) > 0
        else 1.0
    )

    document_frequencies = Counter()

    for tokens in document_tokens:
        document_frequencies.update(set(tokens))

    total_documents = len(document_tokens)
    scores = np.zeros(total_documents, dtype=float)

    query_frequency = Counter(query_tokens)

    for index, tokens in enumerate(document_tokens):
        term_frequencies = Counter(tokens)
        document_length = document_lengths[index]

        for term, query_count in query_frequency.items():
            frequency = term_frequencies.get(term, 0)

            if frequency == 0:
                continue

            doc_frequency = document_frequencies[term]

            inverse_document_frequency = log(
                1 + (
                    total_documents - doc_frequency + 0.5
                ) / (
                    doc_frequency + 0.5
                )
            )

            denominator = frequency + k1 * (
                1 - b + b * document_length / average_length
            )

            term_score = (
                inverse_document_frequency
                * frequency * (k1 + 1)
                / denominator
            )

            scores[index] += query_count * term_score

    return scores


def search_transactions(
    query,
    top_k=3,
    branch=None,
    transaction_type=None,
    minimum_score=0.40,
    query_embedding=None,
    minimum_amount=None,
    maximum_amount=None,
):
    """
    Retrieve transactions using hybrid semantic and BM25 retrieval.

    Result fields:
        index: Original transaction index.
        text: Stored transaction text.
        score: Combined hybrid score.
        semantic_score: Cosine similarity.
        keyword_score: Normalized BM25 score.
    """

    # 1. Validate query
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string")

    query = query.strip()

    # 2. Validate top_k
    if (
        isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or top_k < 1
    ):
        raise ValueError("top_k must be a positive integer")

    # 3. Validate semantic threshold
    if (
        isinstance(minimum_score, bool)
        or not isinstance(minimum_score, (int, float))
        or not np.isfinite(minimum_score)
        or not -1 <= minimum_score <= 1
    ):
        raise ValueError("minimum_score must be between -1 and 1")

    # 4. Validate filters
    _validate_filter(branch, "branch")
    _validate_filter(transaction_type, "transaction_type")
    _validate_amount(minimum_amount, "minimum_amount")
    _validate_amount(maximum_amount, "maximum_amount")

    if (
        minimum_amount is not None
        and maximum_amount is not None
        and minimum_amount > maximum_amount
    ):
        raise ValueError(
            "minimum_amount cannot be greater than maximum_amount"
        )

    # 5. Load corpus
    texts, embeddings = load_embeddings()
    embeddings = np.asarray(embeddings, dtype=float)

    if embeddings.ndim != 2:
        raise ValueError("Stored embeddings must be a 2D array")

    if len(texts) != len(embeddings):
        raise ValueError(
            "Number of transaction texts and embeddings must match"
        )

    if not np.all(np.isfinite(embeddings)):
        raise ValueError("Stored embeddings contain invalid values")

    if len(texts) == 0:
        logger.warning("No transaction data available")
        return []

    # 6. Validate stored vectors
    embedding_norms = np.linalg.norm(embeddings, axis=1)

    if np.any(embedding_norms == 0):
        raise ValueError("Stored embeddings contain zero vectors")

    # 7. Generate or reuse query embedding
    if query_embedding is None:
        query_embedding = create_embeddings([query])[0]

    query_embedding, query_norm = _validate_query_embedding(
        query_embedding,
        embeddings.shape[1],
    )

    # 8. Semantic similarity
    similarities = (
        embeddings @ query_embedding
    ) / (
        embedding_norms * query_norm
    )

    # 9. Apply strict metadata filters
    matching_indices = []

    normalized_branch = branch.casefold() if branch else None
    normalized_type = (
        transaction_type.casefold()
        if transaction_type else None
    )

    for index, text in enumerate(texts):

        if normalized_branch is not None:
            current_branch = _extract_field(text, "Branch")

            if (
                current_branch is None
                or current_branch.casefold() != normalized_branch
            ):
                continue

        if normalized_type is not None:
            current_type = _extract_field(text, "Type")

            if (
                current_type is None
                or current_type.casefold() != normalized_type
            ):
                continue

        if minimum_amount is not None or maximum_amount is not None:
            amount = _extract_amount(text)

            if amount is None:
                continue

            if minimum_amount is not None and amount < minimum_amount:
                continue

            if maximum_amount is not None and amount > maximum_amount:
                continue

        matching_indices.append(index)

    if not matching_indices:
        return []

    # 10. BM25 keyword scores
    filtered_texts = [texts[i] for i in matching_indices]

    keyword_scores = _calculate_bm25(query, filtered_texts)

    maximum_keyword_score = (
        float(np.max(keyword_scores))
        if len(keyword_scores) else 0.0
    )

    if maximum_keyword_score > 0:
        normalized_keyword_scores = (
            keyword_scores / maximum_keyword_score
        )
    else:
        normalized_keyword_scores = np.zeros(
            len(matching_indices),
            dtype=float,
        )

    # 11. Hybrid ranking
    results = []

    for position, index in enumerate(matching_indices):
        semantic_score = float(similarities[index])
        keyword_score = float(normalized_keyword_scores[position])

        # Keep semantic threshold behavior, but allow
        # documents with a genuine keyword match.
        if (
            semantic_score < minimum_score
            and keyword_score == 0
        ):
            continue

        normalized_semantic_score = (
            semantic_score + 1.0
        ) / 2.0

        hybrid_score = (
            SEMANTIC_WEIGHT * normalized_semantic_score
            + KEYWORD_WEIGHT * keyword_score
        )

        results.append({
            "index": index,
            "text": texts[index],
            "score": float(hybrid_score),
            "semantic_score": semantic_score,
            "keyword_score": keyword_score,
        })

    # 12. Sort by hybrid score
    results.sort(
        key=lambda result: result["score"],
        reverse=True,
    )

    logger.info(
        f"Hybrid search completed for '{query}'; "
        f"returned {min(len(results), top_k)} results"
    )

    return results[:top_k]