
from src.semantic_search import search_transactions
from src.retrieval_evaluation import evaluate_dataset
from src.embedding_store import load_embeddings


# Ground-truth labels verified against the four actual transactions.
# Indices:
# 0 = Karachi Deposit 50000
# 1 = Lahore Withdrawal 20000
# 2 = Karachi Withdrawal 90000
# 3 = Islamabad Deposit 15000

EVALUATION_DATASET = [
    {
        "query": "Karachi withdrawal",
        "relevant_indices": {2},
    },
    {
        "query": "Karachi deposit",
        "relevant_indices": {0},
    },
    {
        "query": "Lahore withdrawal",
        "relevant_indices": {1},
    },
    {
        "query": "Islamabad deposit",
        "relevant_indices": {3},
    },
    {
        "query": "all withdrawals",
        "relevant_indices": {1, 2},
    },
    {
        "query": "all deposits",
        "relevant_indices": {0, 3},
    },
    {
        "query": "transactions in Karachi",
        "relevant_indices": {0, 2},
    },
    {
        "query": "transactions above 40000",
        "relevant_indices": {0, 2},
    },
]


def search_baseline(query, top_k=3):
    """Run the existing semantic search without metadata filters."""
    return search_transactions(
        query=query,
        top_k=top_k,
        minimum_score=-1.0,
    )


def run_evaluation():
    texts, embeddings = load_embeddings()

    # Validate benchmark labels against the actual corpus.
    corpus_size = len(texts)

    for item in EVALUATION_DATASET:
        invalid_indices = {
            index
            for index in item["relevant_indices"]
            if not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or index >= corpus_size
        }

        if invalid_indices:
            raise ValueError(
                f"Invalid relevance labels for query "
                f"'{item['query']}': {invalid_indices}"
            )

    metrics = evaluate_dataset(
        evaluation_dataset=EVALUATION_DATASET,
        search_function=search_baseline,
        k=3,
    )

    print("\n=== Semantic Search Benchmark ===")
    print(f"Corpus size: {len(texts)} transactions")
    print(f"Number of queries: {len(EVALUATION_DATASET)}")
    print("Retrieval: unfiltered semantic search")
    print("K = 3")

    print(f"\nMacro Precision@3: {metrics['precision_at_k']:.3f}")
    print(f"Macro Recall@3:    {metrics['recall_at_k']:.3f}")
    print(f"MRR:                {metrics['mrr']:.3f}")

    print("\n=== Per-query Results ===")

    for result, item in zip(
        metrics["per_query"],
        EVALUATION_DATASET,
    ):
        print(f"\nQuery: {result['query']}")
        print(f"Relevant indices: {sorted(item['relevant_indices'])}")
        print(f"Retrieved indices: {result['retrieved_indices']}")
        print(f"Precision@3: {result['precision_at_k']:.3f}")
        print(f"Recall@3: {result['recall_at_k']:.3f}")
        print(f"Reciprocal Rank: {result['reciprocal_rank']:.3f}")

        print("Retrieved transactions:")
        for index in result["retrieved_indices"]:
            print(f"  - {texts[index]}")


if __name__ == "__main__":
    run_evaluation()
