from pathlib import Path
import pytest
from src.data_loader import load_excel

def test_loader_excel_success():
    """
    Test loading a valid Escel file
    """
    
    file_path = Path("E:/AI-Financial-Report-Analyzer/data/transactions.xlsx")
    df = load_excel(file_path)
    assert not df.empty
    assert len(df)==4

def test_file_not_found():
    """
    Test loading an non existing file
    """
    with pytest.raises(FileNotFoundError):
        load_excel(Path("data/not_found.xlsx")) 

def test_invalid_extensions(tmp_path):
    """
    Test invalid file extension.
    """
    file=tmp_path / "sample.csv"
    file.write_text("name,amount")
    
    with pytest.raises(ValueError):
        load_excel(file)