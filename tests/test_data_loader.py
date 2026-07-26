"""Unit tests for src/data_loader.py"""

import pandas as pd
import pytest

from src.data_loader import clean_financial_columns, get_data_summary, load_insurance_data


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "PolicyID": [1, 2, 3, 4, 5],
        "TotalPremium": [1000.0, 2000.0, -50.0, 1500.0, 0.0],
        "TotalClaims": [500.0, 0.0, 100.0, -20.0, 200.0],
    })


def test_load_insurance_data_missing_file_returns_empty_df():
    result = load_insurance_data("this/path/does/not/exist.txt")
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_load_insurance_data_reads_pipe_separated(tmp_path):
    filepath = tmp_path / "sample.txt"
    filepath.write_text("PolicyID|TotalPremium|TotalClaims\n1|1000|500\n2|2000|0\n")
    result = load_insurance_data(str(filepath))
    assert len(result) == 2
    assert list(result.columns) == ["PolicyID", "TotalPremium", "TotalClaims"]


def test_load_insurance_data_respects_nrows(tmp_path):
    filepath = tmp_path / "sample.txt"
    filepath.write_text("PolicyID|TotalPremium|TotalClaims\n1|1000|500\n2|2000|0\n3|1500|100\n")
    result = load_insurance_data(str(filepath), nrows=2)
    assert len(result) == 2


def test_get_data_summary_basic_shape(sample_df):
    summary = get_data_summary(sample_df)
    assert summary["rows"] == 5
    assert summary["columns"] == 3
    assert summary["total_missing"] == 0


def test_get_data_summary_counts_missing_values():
    df = pd.DataFrame({"A": [1, None, 3], "B": [None, None, 6]})
    summary = get_data_summary(df)
    assert summary["missing_columns"] == 2
    assert summary["total_missing"] == 3


def test_get_data_summary_empty_df_returns_dict_without_error():
    summary = get_data_summary(pd.DataFrame())
    assert isinstance(summary, dict)


def test_clean_financial_columns_removes_negative_premium(sample_df):
    cleaned = clean_financial_columns(sample_df)
    assert (cleaned["TotalPremium"] >= 0).all()


def test_clean_financial_columns_removes_negative_claims(sample_df):
    cleaned = clean_financial_columns(sample_df)
    assert (cleaned["TotalClaims"] >= 0).all()


def test_clean_financial_columns_adds_loss_ratio_and_margin(sample_df):
    cleaned = clean_financial_columns(sample_df)
    assert "LossRatio" in cleaned.columns
    assert "Margin" in cleaned.columns


def test_clean_financial_columns_loss_ratio_correct_for_known_row(sample_df):
    cleaned = clean_financial_columns(sample_df)
    # PolicyID 1: premium=1000, claims=500 -> loss ratio 0.5
    row = cleaned[cleaned["PolicyID"] == 1].iloc[0]
    assert row["LossRatio"] == pytest.approx(0.5)


def test_clean_financial_columns_zero_premium_gives_zero_loss_ratio():
    df = pd.DataFrame({"PolicyID": [1], "TotalPremium": [0.0], "TotalClaims": [0.0]})
    cleaned = clean_financial_columns(df)
    assert cleaned.iloc[0]["LossRatio"] == 0.0
