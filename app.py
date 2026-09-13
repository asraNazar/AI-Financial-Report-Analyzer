from pathlib import Path

from src.data_loader import load_excel
from src.validator import validate_dataframe
from src.cleaner import clean_dataFrame
from src.analyzer import analyze_transactions
from src.charts import (create_branch_chart,create_transaction_pie_chart)
from config import DATA_DIR
from src.ai_analyzer import generate_financial_insights
from src.report_generator import generate_financial_report
from src.rag import ask_rag

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

    print("\n" + "=" * 50)
    print("ASK FINANCIAL QUESTION")
    print("=" * 50)

    print("Ask multiple questions about your transactions.")
    print("Type 'exit' to close the assistant.")

    while True:
        query = input("\nEnter your question: ").strip()

        if query.lower() == "exit":
            print("\nExiting financial assistant. Goodbye!")
            break

        if not query:
            print("Please enter a question.")
            continue

        try:
            rag_answer = ask_rag(query,df=df)

            print("\n" + "=" * 50)
            print("RAG RESULT")
            print("=" * 50)
            print(rag_answer)

        except Exception as error:
            print("\nUnable to process your question.")
            print(f"Error: {error}")

if __name__ == "__main__":
    main()