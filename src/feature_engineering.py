import os
import pandas as pd
import numpy as np
from .config import DAILY_PRICES_PATH, ENGINEERED_FEATURES_PATH, SHORT_WINDOW, LONG_WINDOW
from .utils import setup_logging

logger = setup_logging()

def engineer_features(
    daily_prices_path: str = DAILY_PRICES_PATH,
    output_path: str = ENGINEERED_FEATURES_PATH,
    short_window: int = SHORT_WINDOW,
    long_window: int = LONG_WINDOW
) -> pd.DataFrame:
    """
    Add rolling-based features for each product:
    rolling mean, std, z-score, pct change, volatility CV.
    """
    if not os.path.exists(daily_prices_path):
        raise FileNotFoundError(f"Daily prices parquet not found: {daily_prices_path}")

    df = pd.read_parquet(daily_prices_path)
    df = df.sort_values(["product_code", "order_date"])

    def apply_features(group: pd.DataFrame) -> pd.DataFrame:
        group = group.set_index("order_date")

        group["rolling_mean_short"] = (
            group["daily_mean_price"].rolling(window=short_window, min_periods=1).mean()
        )
        group["rolling_mean_long"] = (
            group["daily_mean_price"].rolling(window=long_window, min_periods=1).mean()
        )
        group["rolling_std_long"] = (
            group["daily_mean_price"].rolling(window=long_window, min_periods=1).std()
        )

        # Rolling z-score with safe division
        group["rolling_z_score"] = (
            (group["daily_mean_price"] - group["rolling_mean_long"])
            / group["rolling_std_long"].replace(0, np.nan)
        )

        # Daily percentage change
        group["daily_delta_pct"] = group["daily_mean_price"].pct_change() * 100.0

        # Volatility CV
        group["volatility_cv"] = (
            group["rolling_std_long"] / group["rolling_mean_long"].replace(0, np.nan)
        )

        return group.reset_index()

    featured_df = df.groupby("product_code", group_keys=False).apply(apply_features)
    featured_df.to_parquet(output_path, index=False)
    logger.info(f"Engineered features saved to {output_path}. Shape: {featured_df.shape}")

    return featured_df
