import os
import pandas as pd
from .config import CLEANED_CSV_PATH, DAILY_PRICES_PATH, MIN_HISTORY_DAYS
from .utils import setup_logging

logger = setup_logging()

def aggregate_time_series(
    cleaned_path: str = CLEANED_CSV_PATH,
    output_path: str = DAILY_PRICES_PATH,
    min_days: int = MIN_HISTORY_DAYS
) -> pd.DataFrame:
    """
    Group by product and resample to daily mean price.
    """
    if not os.path.exists(cleaned_path):
        raise FileNotFoundError(f"Cleaned data not found: {cleaned_path}")

    df = pd.read_csv(cleaned_path)
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])

    df = df.sort_values(["product_code", "order_date"])

    grouper = [pd.Grouper(key="order_date", freq="D"), "product_code"]

    aggregated = (
        df.groupby(grouper)["price"]
        .mean()
        .groupby(level="product_code")
        .ffill()
        .reset_index()
        .rename(columns={"price": "daily_mean_price"})
    )

    product_day_counts = aggregated.groupby("product_code")["order_date"].nunique()
    valid_products = product_day_counts[product_day_counts >= min_days].index
    aggregated = aggregated[aggregated["product_code"].isin(valid_products)]

    logger.info(f"Aggregated {len(valid_products)} products with >= {min_days} days")
    logger.info(f"Aggregated shape: {aggregated.shape}")

    aggregated.to_parquet(output_path, index=False)
    logger.info(f"Saved daily prices to {output_path}")

    return aggregated
