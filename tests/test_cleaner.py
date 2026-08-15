import pandas as pd
from src.cleaner import clean_dataFrame

def test_cleaner_branch_name():
    df=pd.DataFrame({
        "Date":["2026-07-26"],
        "Branch":[" karachi "],
        "Type":[" deposit "],
        "Amount":[5000]
    })

    cleaned = clean_dataFrame(df)
    assert cleaned.loc[0,"Branch"] == "Karachi"
    assert cleaned.loc[0,"Type"] == "Deposit"

def test_amount_conversion():
     df=pd.DataFrame({
        "Date":["2026-07-26"],
        "Branch":["Karachi"],
        "Type":["Deposit"],
        "Amount":[5000]
    })
     
     cleaned = clean_dataFrame(df)
     assert cleaned["Amount"].dtype.kind in "i","f"


def text_date_conversion():
     df=pd.DataFrame({
        "Date":["2026-07-26"],
        "Branch":["Karachi"],
        "Type":["Deposit"],
        "Amount":[5000]
    })
     
     cleaned=clean_dataFrame(df)
     assert pd.api.types.is_datetime64_any_dtype(
          cleaned["Date"]
     )
     