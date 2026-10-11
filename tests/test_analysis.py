"""Basic sanity checks for the sales-analysis-prediction project."""
import numpy as np
import pandas as pd
import pytest

CSV = "data/processed/cleaned_sales.csv"


def test_data_counts():
    df = pd.read_csv(CSV)
    assert len(df) > 5000
    assert df["units_sold"].gt(0).all()
    assert (df["discount_percent"] >= 0).all()
    assert df["gross_profit"].notna().all()


def test_summary_file():
    summary = pd.read_csv("model/summary.csv")
    assert "revenue" in summary.iloc[0]["metric"].lower()
