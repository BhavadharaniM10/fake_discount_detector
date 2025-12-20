# tests/test_detection.py
import pytest
import pandas as pd
import numpy as np
from datetime import timedelta
from src.rule_detection import detect_fake_discount

# -----------------------------
# Helpers
# -----------------------------
def make_mock_engineered_df():
    """Create a synthetic dataset with spikes and drops for testing."""
    dates = pd.date_range("2021-01-01", periods=100)
    prices = np.random.normal(50, 5, 100)

    # Inject artificial spike and drop
    prices[80:85] = 80   # spike
    prices[95] = 40      # drop

    df = pd.DataFrame({
        "product_code": ["TEST"] * len(dates),
        "order_date": dates,
        "daily_mean_price": prices,
    })

    df["rolling_mean_long"] = df["daily_mean_price"].rolling(30, min_periods=1).mean()
    df["rolling_std_long"] = df["daily_mean_price"].rolling(30, min_periods=1).std()
    df["rolling_z_score"] = (
        (df["daily_mean_price"] - df["rolling_mean_long"])
        / df["rolling_std_long"].replace(0, np.nan)
    )
    df["volatility_cv"] = df["rolling_std_long"] / df["rolling_mean_long"].replace(0, np.nan)

    return df

# -----------------------------
# Tests
# -----------------------------

def test_detect_fake_discount_basic():
    """Smoke test: function runs and returns expected keys."""
    df = make_mock_engineered_df()
    eval_date = df["order_date"].max().strftime("%Y-%m-%d")
    result = detect_fake_discount(df, "TEST", eval_date)
    assert "discount_status" in result
    assert "explanation" in result
    assert "volatility_score" in result

def test_detect_fake_discount_suspicious():
    """Check detection around spike region."""
    df = make_mock_engineered_df()
    eval_date = df["order_date"].iloc[82].strftime("%Y-%m-%d")
    result = detect_fake_discount(df, "TEST", eval_date)
    assert result["discount_status"] in ["Suspicious", "Genuine"]

def test_detect_fake_discount_anchor_date():
    """Ensure anchoring works when date not in dataset."""
    df = make_mock_engineered_df()
    eval_date = "2025-01-01"  # future date not in dataset
    result = detect_fake_discount(df, "TEST", eval_date)
    # Should anchor to last available date
    assert result["evaluation_date"] == str(df["order_date"].max().date())

def test_detect_fake_discount_insufficient_data():
    """Check behavior when not enough records in window."""
    df = make_mock_engineered_df().iloc[:3]  # only 3 records
    eval_date = df["order_date"].max().strftime("%Y-%m-%d")
    result = detect_fake_discount(df, "TEST", eval_date)
    assert result["discount_status"] == "Insufficient Data"

def test_detect_fake_discount_no_discount():
    """Real-time mode: claimed original < current price."""
    df = make_mock_engineered_df()
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    result = detect_fake_discount(
        df,
        "TEST",
        today,
        current_price=100,
        claimed_original_price=90,
    )
    assert result["discount_status"] == "No Discount"

def test_detect_fake_discount_invalid_product():
    """Invalid product code should return No Data."""
    df = make_mock_engineered_df()
    eval_date = df["order_date"].max().strftime("%Y-%m-%d")
    result = detect_fake_discount(df, "INVALID", eval_date)
    assert result["discount_status"] == "No Data"
