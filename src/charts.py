import matplotlib.pyplot as plt
from config import CHARTS_DIR
from src.logger import logger

def create_branch_chart(branch_summary):
    """
    Create a bar chart of branch-wise transaction amounts.
    """
    logger.info("Create branch summary chart")
    try:
        plt.figure(figsize=(8,5))
        plt.bar(
            branch_summary.index,
            branch_summary.values)
        plt.title("Branch wise transaction amount")
        plt.xlabel("Branch")
        plt.ylabel("Amount")
        plt.tight_layout()
        chart_path = CHARTS_DIR / "branch_summary.png"
        plt.savefig(chart_path)
        plt.close()
        logger.info(f"Chart Saved {chart_path}")

    except Exception as error:
        logger.error(f"Error Creating chart: {error}")
        raise

def create_transaction_pie_chart(total_deposits,total_withdrawals):
    """
    Create a pie chart showing deposits vs withdrawals.
    """
    logger.info("Creating transaction type pie chart")
    try:
        plt.figure(figsize=(6,6))
        plt.pie(
            [total_deposits,total_withdrawals],
            labels=["Deposits","Wihdrawals"],
                    autopct="%1.1f%%",
                    startangle=90        )
        plt.title("Deposit vs Withdraws")
        charts_path=CHARTS_DIR/"transaction_types.png"
        plt.savefig(charts_path)
        plt.close()
        logger.info(f"Chart saved {charts_path}") 
    except Exception as error:
        logger.error(f"Error creating transaction pie chart: {error}")  
        raise       
                                 
