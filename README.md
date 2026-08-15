# AI Financial Report Analyzer

A Python-based application that analyzes financial transaction data from Excel files and generates financial reports.

## Features

- Load transaction data from Excel
- Validate financial data
- Clean transaction data
- Analyze transactions
- Calculate total transactions and amounts
- Calculate deposits and withdrawals
- Find highest and lowest transactions
- Analyze branch-wise performance
- Identify most and least active branches
- Generate branch-wise charts
- Generate deposit vs withdrawal pie chart
- Generate AI-powered financial insights
- Generate PDF financial reports
- Automated testing with Pytest
- Error handling and logging

## Tech Stack

- Python
- Pandas
- OpenPyXL
- Matplotlib
- ReportLab
- OpenAI API
- Python-dotenv
- Loguru
- Pytest

## Project Structure

```text
AI-Financial-Report-Analyzer/
│
├── data/
├── charts/
├── reports/
├── src/
├── tests/
│
├── app.py
├── config.py
├── pytest.ini
├── requirements.txt
├── .gitignore
└── README.md


## How to Run

### 1. Create a virtual environment

```powershell
python -m venv .venv

.venv\Scripts\Activate.ps1

(.venv) PS E:\AI-Financial-Report-Analyzer>

pip install -r requirements.txt

OPENAI_API_KEY=your_api_key_here

python app.py

The application will:

Load the Excel transaction file.
Validate the data.
Clean the data.
Analyze transactions.
Generate AI financial insights.
Generate charts.
Generate the PDF financial report.

Run the tests
pytest -v