from unittest.mock import patch,MagicMock
from src.ai_analyzer import generate_financial_insights

def test_ai_insights_without_api_key():
    results={}
    with patch("src.ai_analyzer.OPENAI_API_KEY",None):
         result=generate_financial_insights(results)
    assert "API key is not configured" in result

def test_ai_insights_success():
     results={
          "total_transactions": 4,
        "total_amount": 175000,
        "total_deposits": 65000,
        "total_withdrawals": 110000,
        "highest_transactions": 90000,
        "lowest_transactions": 15000,
        "branch_summary": "Karachi 140000, Lahore 20000, Islamabad 15000",
        "most_active_branch": "Karachi",
        "least_active_branch": "Islamabad",
     }

     mock_response = MagicMock()
     mock_response.output_text=(
          "Karachi is the most active branch and withdrawals "
        "are higher than deposits."
     )
     with patch("src.ai_analyzer.OPENAI_API_KEY","fake_key"):
          with patch("src.ai_analyzer.OpenAI") as mock_openai:
               mock_client = mock_openai.return_value
               mock_client.responses.create.return_value=mock_response
               result=generate_financial_insights(results)
     assert "Karachi" in result
     mock_client.responses.create.assert_called_once()   
