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
    current_price: float = None,
    claimed_original_price: float = None,
    window_days: int = RECENT_WINDOW_DAYS,
    drop_threshold: float = DROP_THRESHOLD,
    spike_z_threshold: float = SPIKE_Z_THRESHOLD,
) -> dict:
    """
    Rule-based fake discount detection.
    - Historical mode: dataset price (anchored to closest previous available date if exact date missing)
    - Real-time mode: user-entered price for today, anchored to last dataset window
    """

    eval_date = pd.to_datetime(evaluation_date).normalize()
    today = pd.Timestamp.today().normalize()

    product_df = df[df["product_code"] == product_code].set_index("order_date").sort_index()
    if product_df.empty:
        return {
            "product_code": product_code,
            "evaluation_date": str(eval_date.date()),
            "discount_status": "No Data",
            "volatility_score": 0.0,
            "explanation": "No price data available for this product.",
        }

    # -------------------------------
    # HISTORICAL MODE (updated logic)
    # -------------------------------
    if current_price is None and claimed_original_price is None:
        # If exact date not present, pick the closest previous date with data
        if eval_date not in product_df.index.normalize():
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
        start_date = eval_date - timedelta(days=window_days)
        recent = product_df.loc[start_date:eval_date]

        if len(recent) < 5:
            return {
                "product_code": product_code,
                "evaluation_date": str(eval_date.date()),
                "discount_status": "Insufficient Data",
                "volatility_score": 0.0,
                "explanation": "Not enough recent historical data for reliable assessment.",
            }

        claimed_original_price = recent["daily_mean_price"].max()

    # -------------------------------
    # REAL-TIME MODE (unchanged)
    # -------------------------------
    elif eval_date == today:
        if current_price is None or claimed_original_price is None or current_price <= 0:
            return {
                "product_code": product_code,
                "evaluation_date": str(eval_date.date()),
                "discount_status": "Invalid Input",
                "explanation": "Please provide valid current and claimed original prices.",
            }

        if claimed_original_price < current_price:
            return {
                "product_code": product_code,
                "evaluation_date": str(eval_date.date()),
                "current_price": round(float(current_price), 2),
                "claimed_original_price": round(float(claimed_original_price), 2),
                "drop_percentage": 0.0,
                "discount_status": "No Discount",
                "volatility_score": 0.0,
                "explanation": "Claimed original price is less than current price. No discount detected.",
            }

        last_date = product_df.index.max()
        start_date = last_date - timedelta(days=window_days)
        recent = product_df.loc[start_date:last_date]

        if len(recent) < 3:  # relaxed threshold for today mode
            return {
                "product_code": product_code,
                "evaluation_date": str(eval_date.date()),
                "discount_status": "Limited Data",
                "volatility_score": 0.0,
                "explanation": (
                    f"Dataset ends on {last_date.date()} with only {len(recent)} usable records "
                    f"in the recent window. Today's evaluation is anchored to that date, so results are approximate."
                ),
            }

    # -------------------------------
    # COMMON LOGIC
    # -------------------------------
    drop_pct = (claimed_original_price - current_price) / claimed_original_price if claimed_original_price > 0 else 0.0
    pre_drop = recent.iloc[:-1]

    has_spike = "rolling_z_score" in pre_drop.columns and (pre_drop["rolling_z_score"] > spike_z_threshold).any()
    vol_normalized = 0.0
    if "volatility_cv" in pre_drop.columns and not pre_drop["volatility_cv"].dropna().empty:
        vol_raw = pre_drop["volatility_cv"].iloc[-1]
        vol_normalized = float(np.clip(vol_raw / 0.35, 0.0, 1.0))

    if drop_pct < drop_threshold:
        status = "Genuine"
        explanation = f"Genuine: Minor drop of {drop_pct:.1%} from recent high."
    elif has_spike:
        status = "Suspicious"
        explanation = f"Suspicious: Sharp drop of {drop_pct:.1%} after price spike detected. Volatility score: {vol_normalized:.2f}."
    else:
        status = "Genuine"
        explanation = f"Genuine: Drop of {drop_pct:.1%} with no anomalies. Volatility score: {vol_normalized:.2f}."

    return {
        "product_code": product_code,
        "evaluation_date": str(eval_date.date()),
        "current_price": round(float(current_price), 2),
        "claimed_original_price": round(float(claimed_original_price), 2),
        "drop_percentage": round(float(drop_pct), 3),
        "discount_status": status,
        "volatility_score": vol_normalized,
        "explanation": explanation,
    }


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
