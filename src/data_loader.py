from pathlib import Path
import pandas as pd
from typing import Union
from src.logger import logger


def load_excel(file_path: Union[str, Path]) -> pd.DataFrame:
    file_path = Path(file_path)
    """
    Load transaction data from an Excel file.

    Args:
        file_path: Path of Excel file.

    Returns:
        Pandas DataFrame containing transaction data.

    Raises:
        FileNotFoundError: If file does not exist.
        Exception: For unexpected errors.
    """

    try:
        logger.info(f"Loading file: {file_path}")

        if not file_path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        if file_path.suffix.lower() not in [".xlsx", ".xls"]:
            raise ValueError(
                "Only Excel files are supported."
            )

        df = pd.read_excel(file_path)

        logger.info(
            f"File loaded successfully. Rows: {len(df)}"
        )

        return df

    except Exception as error:
        logger.error(
            f"Error loading file: {error}"
        )

        raise