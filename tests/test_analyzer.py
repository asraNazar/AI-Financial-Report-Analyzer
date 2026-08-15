import pandas as pd
from src.analyzer import analyze_transactions

def test_analyze_transactions():
    df = pd.DataFrame(
        {
            "Date": [
                "2026-01-01",
                "2026-01-02",
                "2026-01-03",
                "2026-01-04",
            ],
            "Branch":[
                "Karachi",
                "Lahore",
                "Karachi",
                "Islamabad",
            ],
            "Type":[
                "Deposit",
                "Withdrawal",
                "Withdrawal",
                "Deposit",
            ],
            "Amount":[
                50000,
                20000,
                90000,
                15000,
            ]
        }
    )
    results = analyze_transactions(df)
    assert results["total_transactions"] == 4
    assert results["total_amount"] ==175000
    assert results["total_deposits"]==65000
    assert results["total_withdrawals"]==110000
    assert results["highest_transactions"]==90000
    assert results["lowest_transactions"] ==15000
    assert results["most_active_branch"]=="Karachi"
    assert results["least_active_branch"]=="Islamabad"