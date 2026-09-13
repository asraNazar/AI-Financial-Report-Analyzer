import json
from pathlib import Path

import numpy as np

from src.logger import logger


EMBEDDINGS_FILE = Path("data/embeddings.npy")
TEXTS_FILE = Path("data/embedding_texts.json")


def save_embeddings(texts, embeddings):
    """
    Save transaction texts and their embeddings locally.
    """
    if len(texts) != len(embeddings):
        raise ValueError(
            "Number of texts must match number of embeddings"
        )

    EMBEDDINGS_FILE.parent.mkdir(parents=True, exist_ok=True)

    np.save(EMBEDDINGS_FILE, embeddings)

    with open(TEXTS_FILE, "w", encoding="utf-8") as file:
        json.dump(texts, file, indent=4, ensure_ascii=False)

    logger.info(
        f"Saved {len(texts)} embeddings to {EMBEDDINGS_FILE}"
    )


def load_embeddings():
    """
    Load saved transaction texts and embeddings.
    """
    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(
            f"Embeddings file not found: {EMBEDDINGS_FILE}"
        )

    if not TEXTS_FILE.exists():
        raise FileNotFoundError(
            f"Texts file not found: {TEXTS_FILE}"
        )

    embeddings = np.load(EMBEDDINGS_FILE)

    with open(TEXTS_FILE, "r", encoding="utf-8") as file:
        texts = json.load(file)

    logger.info(
        f"Loaded {len(texts)} embeddings successfully"
    )

    return texts, embeddings