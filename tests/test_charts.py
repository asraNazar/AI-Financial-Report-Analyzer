from pathlib import Path
import pandas as pd

from src.charts import (create_branch_chart,create_transaction_pie_chart)

def test_create_pie_chart():
    
    branch_summary = pd.Series(
        [140000, 20000, 15000],
        index=["Karachi", "Lahore", "Islamabad"]
    )
    create_branch_chart(branch_summary)
    charts_path = Path("charts/branch_summary.png")
    assert charts_path.exists

def test_create_transaction_pie_chart():
    create_transaction_pie_chart(
        65000,110000
    ) 
    chart_path=Path("chart/transactions_types.png")   
    assert chart_path.exists