from typing import List
import pandas as pd
from src.logger import logger
from config import REQUIRED_COLUMNS,COLUMN_MAPPING

def validate_dataframe(df: pd.DataFrame) -> bool:
    df.rename(columns=COLUMN_MAPPING,inplace=True)
    missing_columns=[column 
                     for column in REQUIRED_COLUMNS
                     if column not in df.columns]
    """
    Validate transaction data.

    Args:
        df: Transaction DataFrame.

    Returns:
        True if validation passes.

    Raises:
        ValueError: If validation fails.
    """

    logger.info("Starting data validation...")

    # Check if DataFrame is empty
    if df.empty:
        logger.error("Excel file is empty.")
        raise ValueError("Excel file contains no data.")

    # Check required columns
    missing_columns: List[str] = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        logger.error(
            f"Missing columns: {missing_columns}"
        )
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Check duplicate column names
    if df.columns.duplicated().any():
        logger.error("Duplicate column names found.")
        raise ValueError("Duplicate column names detected.")

    # Check Amount column
    if not pd.api.types.is_numeric_dtype(df["Amount"]):
        logger.error("Amount column must be numeric.")
        raise ValueError("Amount column must contain numeric values.")

    # Check Date column
    try:
        pd.to_datetime(df["Date"])
    except Exception:
        logger.error("Invalid dates found.")
        raise ValueError("Date column contains invalid dates.")

    logger.info("Data validation completed successfully.")

    return True