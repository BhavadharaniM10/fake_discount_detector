import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.rule_detection import detect_fake_discount

def make_mock_engineered_df():
    dates = pd.date_range("2021-01-01", periods=100)
    prices = np.random.normal(50, 5, 100)
    prices[80:85] = 80
    prices[95] = 40

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

def test_detect_fake_discount_basic():
    df = make_mock_engineered_df()
    eval_date = df["order_date"].max().strftime("%Y-%m-%d")
    result = detect_fake_discount(df, "TEST", eval_date)

    assert "discount_status" in result
    assert "explanation" in result
    assert "volatility_score" in result
