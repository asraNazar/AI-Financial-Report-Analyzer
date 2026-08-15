from pathlib import Path
from dotenv import load_dotenv
import os


# Load environment variables from .env
load_dotenv()


# Project base directory
BASE_DIR = Path(__file__).resolve().parent


# Project folders
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
CHARTS_DIR = BASE_DIR / "charts"
LOGS_DIR = BASE_DIR / "logs"


# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


REQUIRED_COLUMNS = [
    "Date",
    "Branch",
    "Type",
    "Amount",
]

COLUMN_MAPPING = {
    "Transaction_Type": "Type",
    "Transaction Type": "Type",
    "Txn_Type": "Type",
}

# Create folders if they don't exist
for folder in [
    DATA_DIR,
    REPORTS_DIR,
    CHARTS_DIR,
    LOGS_DIR
]:
    folder.mkdir(exist_ok=True)