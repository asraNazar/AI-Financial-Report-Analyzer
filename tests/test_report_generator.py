from pathlib import Path
from src.report_generator import generate_financial_report

def test_generate_financial_report(tmp_path,monkeypatch):
    results={
        "total_transactions":4,
        "total_amount":175000,
        "total_deposits": 65000,
        "total_withdrawals": 110000,
        "highest_transactions": 90000,
        "lowest_transactions": 15000,
        "branch_summary": {
            "Karachi": 140000,
            "Lahore": 20000,
            "Islamabad": 15000,
        },
        "most_active_branch": "Karachi",
        "least_active_branch": "Islamabad",
    }
    ai_insights=("Karachi is the most active branch. "
        "Withdrawals are higher than deposits.")

    import src.report_generator as report_generator
    monkeypatch.setattr(report_generator,"REPORTS_DIR",tmp_path)
    report_path=generate_financial_report(results,ai_insights)

    assert report_path.exists()
    assert report_path.suffix ==".pdf"
    assert report_path.name =="financial_report.pdf"
    assert report_path.stat().st_size >0