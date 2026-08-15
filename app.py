from pathlib import Path

from src.data_loader import load_excel
from src.validator import validate_dataframe
from src.cleaner import clean_dataFrame
from src.analyzer import analyze_transactions
from src.charts import (create_branch_chart,create_transaction_pie_chart)
from config import DATA_DIR
from src.ai_analyzer import generate_financial_insights
from src.report_generator import generate_financial_report

def main():
    # file_path = Path("E:/AI-Financial-Report-Analyzer/data/transactions.xlsx")
    file_path = DATA_DIR/"transactions.xlsx"
    df = load_excel(file_path)
# print(df.columns.tolist())
    validate_dataframe(df)
    df=clean_dataFrame(df)
    
    results = analyze_transactions(df)
    ai_insights = generate_financial_insights(results)
    create_branch_chart(results[
        "branch_summary"
    ])
    create_transaction_pie_chart(
        results["total_deposits"],
        results["total_withdrawals"]
    )

    report_path = generate_financial_report(
        results,ai_insights
    )
   
    print("\n" + "=" * 50)
    print("FINANCIAL REPORT ANALYSIS")
    print("=" * 50)
    print("\n" + "=" * 50)
    print("AI FINANCIAL INSIGHTS") 
    print("=" * 50)
    print(ai_insights)
    for key,value in results.items():
        print(f"{key}:")
        print(value)

    print(f"\nPDF report generated: {report_path}")
if __name__ == "__main__":
    main()