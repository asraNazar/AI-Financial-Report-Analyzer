import pandas as pd
from src.logger import logger

def analyze_transactions(df:pd.DataFrame) -> dict:
    """"
Analyze transaction data

Args:
 Cleaned transaction dataframe

 Returns:
 Dictionary containing analysis results
    """

    logger.info("Starting transaction analysis")
    total_trans = len(df)
    total_amount = df["Amount"].sum()
    total_deposits = (
        df.loc[df["Type"]=="Deposit","Amount"].sum()

    )

    total_withdrawals = (
        df.loc[df["Type"]=="Withdrawal","Amount"].sum()
    )

    highest_transactions = df["Amount"].max()
    lowest_transations = df["Amount"].min()

    branch_summary = (
        df.groupby("Branch")["Amount"].sum()
        .sort_values(ascending=False)
    )

    most_active_branch = (
        df.groupby("Branch")["Amount"].sum().idxmax()   )

    least_active_branch =(
        df.groupby("Branch")["Amount"].sum().idxmin()
    )

    logger.info("Transaction analysis completed")

    return {
        "total_transactions": total_trans,
        "total_amount":total_amount,
        "total_deposits":total_deposits,
        "total_withdrawals":total_withdrawals,
        "highest_transactions":highest_transactions,
        "lowest_transactions":lowest_transations,
        "branch_summary":branch_summary,
        "most_active_branch":most_active_branch,
        "least_active_branch":least_active_branch
    }
