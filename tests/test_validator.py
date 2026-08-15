import pandas as pd
import pytest

from src.validator import validate_dataframe

def test_validator_success():
    df=pd.DataFrame(
        {
            "Date":["2026-07-26"],
            "Branch": ["Karachi"],
            "Type":["Deposit"],
            "Amount":[5000]
        }
    )
    assert validate_dataframe(df) is True

def test_missing_columns():
    df = pd.DataFrame(
        {
            "Date":["2026-07-26"],
            "Branch":["Karachi"],
            "Amount":[5000]
        }
    )   

    with pytest.raises(ValueError):
        validate_dataframe(df)

def test_empty_dataFrame():
    df= pd.DataFrame()

    with pytest.raises(ValueError) :
        validate_dataframe(df)

def test_amount_not_numeric():
    df=pd.DataFrame({
        "Date":["2026-07-26"],
        "Branch":["Karachi"],
        "Type":["Deposit"],
        "Amount":["Five Thousand"]
    })

    with pytest.raises(ValueError):
        validate_dataframe(df)        