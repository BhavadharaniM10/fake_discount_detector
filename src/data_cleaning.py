import os
import pandas as pd
from .config import RAW_CSV_PATH, CLEANED_CSV_PATH
from .utils import setup_logging

logger = setup_logging()

def load_and_clean_data(
    file_path: str = RAW_CSV_PATH,
    output_path: str = CLEANED_CSV_PATH
) -> pd.DataFrame:
    """
    Load raw dataset, clean it, and save cleaned CSV.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    df = pd.read_csv(file_path)
    logger.info(f"Loaded raw dataset with {len(df)} rows")

    # Rename columns
    rename_map = {
        "Order Date": "order_date",
        "Purchase Price Per Unit": "price",
        "Quantity": "quantity",
        "Shipping Address State": "shipping_state",
        "Title": "title",
        "ASIN/ISBN (Product Code)": "product_code",
        "Category": "category",
        "Survey ResponseID": "survey_id",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Strict numeric conversion
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")

    # Convert date
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

    # Drop invalid critical fields
    initial_rows = len(df)
    df = df.dropna(subset=["order_date", "product_code", "price", "quantity"])
    df = df[(df["price"] > 0) & (df["quantity"] > 0)]
    dropped = initial_rows - len(df)
    logger.warning(f"Dropped {dropped} rows due to invalid data ({dropped / initial_rows:.2%})")

    # Sort and subset columns
    df = df.sort_values(["product_code", "order_date"]).reset_index(drop=True)
    keep_cols = [
        "order_date",
        "price",
        "quantity",
        "product_code",
        "category",
        "title",
        "shipping_state",
    ]
    df = df[[c for c in keep_cols if c in df.columns]]

    # Save cleaned data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Cleaned dataset saved to {output_path}. Shape: {df.shape}")
    logger.info(f"price dtype: {df['price'].dtype}")

    return df
