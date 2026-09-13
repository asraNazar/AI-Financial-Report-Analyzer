from sentence_transformers import SentenceTransformer
from src.logger import logger


MODEL_NAME = "all-MiniLM-L6-v2"

logger.info(f"Loading embedding model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)
logger.info("Embedding model loaded successfully")


def create_embeddings(texts: list[str]):
    """
    Convert a list of text documents into numerical embeddings.

    Args:
        texts: List of text strings.

    Returns:
        NumPy array containing embeddings.
    """
    if not texts:
        raise ValueError("Texts list cannot be empty")

    logger.info(f"Creating embeddings for {len(texts)} texts")

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False
    )

    logger.info(
        f"Embeddings created successfully. Shape: {embeddings.shape}"
    )

    return embeddings


def transactions_to_text(df):
    """
    Convert transaction DataFrame rows into text documents
    suitable for embedding generation.
    """
    required_columns = {"Date", "Branch", "Type", "Amount"}

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError("Transaction DataFrame cannot be empty")

    texts = []

    for _, row in df.iterrows():
        text = (
            f"Date: {row['Date']} | "
            f"Branch: {row['Branch']} | "
            f"Type: {row['Type']} | "
            f"Amount: {row['Amount']}"
        )

        texts.append(text)

    logger.info(
        f"Converted {len(texts)} transactions into text documents"
    )

    return texts