from openai import OpenAI
from config import OPENAI_API_KEY
from src.logger import logger

def generate_financial_insights(results:dict)->str:
    """
    Generate AI-powered financial insights from analysis results.

    If an OpenAI API key is not configured, return a safe message
    instead of making an API request.
    """
    logger.info("Starting AI financial insight generation")
    if not OPENAI_API_KEY:
        logger.warning(
            "OPENAI_KEY not configured."
            "Skipping AI insight generation"
        )
        return (
            "AI insights unavailable:"
            "OpenAI API key is not configured"
        )
    try:
        client=OpenAI(api_key=OPENAI_API_KEY)
        prompt=f"""
You are a financial data analyst.
Analyze the following transaction analysis results:

Total transactions :{results["total_transactions"]}
Total amount: {results["total_amount"]}
Total deposits: {results["total_deposits"]}
Total withdrawals: {results["total_withdrawals"]}
Highest transaction: {results["highest_transactions"]}
Lowest transaction: {results["lowest_transactions"]}
Branch summary: {results["branch_summary"]}
Most active branch: {results["most_active_branch"]}
Least active branch: {results["least_active_branch"]}

Provide a concise financial analysis covering:
1. Overall transaction activity
2. Deposit vs withdrawal behavior
3. Branch performance
4. Any notable observations

Do not invent data that is not provided.
Base your analysis only on the supplied results.
"""
        response = client.responses.create(
            model="gpt-5.5",
            input=prompt
        )
        insights = response.output_text
        logger.info("AI finacial insights generation  completed")
        return insights
    except Exception as error:
        logger.error(f"Error generating ai insights: {error}")
        return(
            "AI insight could not generated"
            " because of an API error"
        )