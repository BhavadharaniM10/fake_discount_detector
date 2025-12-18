from datetime import timedelta
import pandas as pd
import numpy as np
from .config import (
    ENGINEERED_FEATURES_PATH,
    RECENT_WINDOW_DAYS,
    DROP_THRESHOLD,
    SPIKE_Z_THRESHOLD,
)
from .utils import setup_logging

logger = setup_logging()

def load_engineered_features(path: str = ENGINEERED_FEATURES_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


def detect_fake_discount(
    df: pd.DataFrame,
    product_code: str,
    evaluation_date: str,
    window_days: int = RECENT_WINDOW_DAYS,
    drop_threshold: float = DROP_THRESHOLD,
    spike_z_threshold: float = SPIKE_Z_THRESHOLD,
) -> dict:
    """
    Rule-based fake discount detection using historical price, z-score spikes, and volatility.
    Uses the closest previous available date if the exact evaluation date has no data.
    """
    eval_date = pd.to_datetime(evaluation_date)

    # Filter product history up to evaluation date (inclusive)
    product_df = df[df["product_code"] == product_code].set_index("order_date")
    product_df = product_df.sort_index()

    if product_df.empty:
        return {
            "product_code": product_code,
            "evaluation_date": str(eval_date.date()),
            "discount_status": "No Data",
            "volatility_score": 0.0,
            "explanation": "No price data available for this product.",
        }

    # If exact date not present, pick the closest previous date with data
    if eval_date not in product_df.index:
        available_dates = product_df.index[product_df.index <= eval_date]
        if len(available_dates) == 0:
            return {
                "product_code": product_code,
                "evaluation_date": str(eval_date.date()),
                "discount_status": "No Data",
                "volatility_score": 0.0,
                "explanation": "No price data available on or before this date.",
            }
        eval_date = available_dates.max()
        logger.info(
            f"No data on requested evaluation date; using closest previous date {eval_date.date()} instead."
        )

    current_price = product_df.loc[eval_date, "daily_mean_price"]

    # Recent window for inference
    start_date = eval_date - timedelta(days=window_days)
    recent = product_df.loc[start_date:eval_date]

    # Relaxed minimum recent history length
    if len(recent) < 5:
        return {
            "product_code": product_code,
            "evaluation_date": str(eval_date.date()),
            "discount_status": "Insufficient Data",
            "volatility_score": 0.0,
            "explanation": "Not enough recent historical data for reliable assessment.",
        }

    claimed_original = recent["daily_mean_price"].max()
    if claimed_original > 0:
        drop_pct = (claimed_original - current_price) / claimed_original
    else:
        drop_pct = 0.0

    pre_drop = recent.iloc[:-1]  # exclude current day

    has_spike = False
    if "rolling_z_score" in pre_drop.columns:
        has_spike = (pre_drop["rolling_z_score"] > spike_z_threshold).any()

    if "volatility_cv" in pre_drop.columns and not pre_drop["volatility_cv"].dropna().empty:
        vol_raw = pre_drop["volatility_cv"].iloc[-1]
        vol_normalized = float(np.clip(vol_raw / 0.35, 0.0, 1.0))
    else:
        vol_normalized = 0.0

    if drop_pct < drop_threshold:
        status = "Genuine"
        explanation = f"Genuine: Minor drop of {drop_pct:.1%} from recent high."
    elif has_spike:
        status = "Suspicious"
        explanation = (
            f"Suspicious: Sharp drop of {drop_pct:.1%} after price spike detected. "
            f"Volatility score: {vol_normalized:.2f}."
        )
    else:
        status = "Genuine"
        explanation = (
            f"Genuine: Drop of {drop_pct:.1%} with no anomalies. "
            f"Volatility score: {vol_normalized:.2f}."
        )

    result = {
        "product_code": product_code,
        "evaluation_date": str(eval_date.date()),
        "current_price": round(float(current_price), 2),
        "claimed_original_price": round(float(claimed_original), 2),
        "drop_percentage": round(float(drop_pct), 3),
        "discount_status": status,
        "volatility_score": vol_normalized,
        "explanation": explanation,
    }

    return result

def generate_explanation(result: dict) -> str:
    """
    Generate plain-language explanation using detection dict.
    """
    status = result.get("discount_status", "Unknown")
    drop_pct = result.get("drop_percentage", 0.0)
    vol_score = result.get("volatility_score", 0.0)

    if status == "Suspicious":
        return (
            f"The discount appears suspicious. "
            f"There was a sharp drop of {drop_pct * 100:.1f}% from a recent high of "
            f"₹{result.get('claimed_original_price', 0):.2f} to the current price of "
            f"₹{result.get('current_price', 0):.2f}. "
            f"A price spike was detected prior to the drop, and high volatility "
            f"(score: {vol_score:.2f}) suggests potential artificial inflation."
        )


    elif status == "Genuine":
        return (
            f"The discount appears genuine. "
            f"Drop of {drop_pct * 100:.1f}% with no unusual spikes in recent history. "
            f"Volatility score: {vol_score:.2f}."
        )

    else:
        return result.get("explanation", "No explanation available.")
