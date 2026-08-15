import pandas as pd
from src.logger import logger

def clean_dataFrame(df:pd.DataFrame) -> pd.DataFrame:
    """"
    Clean transaction data.

    Args:
         df: Raw transaction dataFrame.
    
    Returns: 
            Clean dataFrame.
    """
    logger.info("Starting Data cleaning....")
    df = df.dropna(how="all")
    df.columns=df.columns.str.strip()

    df["Branch"]= (
        df["Branch"].astype(str)
        .str.strip()
        .str.title()
    )

    df["Type"]=(
        df["Type"]
        .astype(str)
        .str.strip()
        .str.title()
    )

    df["Date"]=pd.to_datetime(df["Date"])
    df["Amount"]=pd.to_numeric(df["Amount"])
    logger.info("Data Cleaning Completed")

    return df