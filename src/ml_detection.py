from datetime import timedelta
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from .config import (
    IFOREST_N_ESTIMATORS,
    IFOREST_CONTAMINATION,
    IFOREST_RANDOM_STATE,
    ENGINEERED_FEATURES_PATH,
    RECENT_WINDOW_DAYS,
    DROP_THRESHOLD,
    SPIKE_Z_THRESHOLD,
)
from .rule_detection import detect_fake_discount
from .utils import setup_logging

logger = setup_logging()


def load_engineered_features(path: str = ENGINEERED_FEATURES_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


def train_isolation_forest(
    df: pd.DataFrame,
    n_estimators: int = IFOREST_N_ESTIMATORS,
    contamination: float = IFOREST_CONTAMINATION,
    random_state: int = IFOREST_RANDOM_STATE,
):
    """
    Train Isolation Forest on flattened price histories.
    Returns model and product -> anomaly score mapping.
    """
    max_length = df.groupby("product_code")["order_date"].count().max()
    X = []
    product_codes = []

    for prod, group in df.groupby("product_code"):
        prices = group.sort_values("order_date")["daily_mean_price"].values
        if len(prices) == 0:
            padded = np.zeros(max_length)
        else:
            padded = np.pad(
                prices,
                (0, max_length - len(prices)),
                mode="constant",
                constant_values=prices[-1],
            )
        X.append(padded)
        product_codes.append(prod)

    X = np.array(X)

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
    )
    model.fit(X)

    scores = model.decision_function(X)
    scores = -scores  # lower = more anomalous, invert

    if scores.max() == scores.min():
        norm_scores = np.zeros_like(scores)
    else:
        norm_scores = (scores - scores.min()) / (scores.max() - scores.min())

    score_dict = dict(zip(product_codes, norm_scores))
    logger.info("Isolation Forest trained for ML anomaly scores.")
    return model, score_dict


def hybrid_detect_fake_discount(
    df: pd.DataFrame,
    score_dict: dict,
    product_code: str,
    evaluation_date: str,
    current_price: float = None,
    claimed_original_price: float = None,
    window_days: int = RECENT_WINDOW_DAYS,
    drop_threshold: float = DROP_THRESHOLD,
    spike_z_threshold: float = SPIKE_Z_THRESHOLD,
) -> dict:
    """
    Hybrid ML + rule-based fake discount detection.
    Supports:
    - Historical mode (dataset price)
    - Real-time mode (user-entered price for today, anchored to last dataset window)
    """

    eval_date = pd.to_datetime(evaluation_date).normalize()
    today = pd.Timestamp.today().normalize()

    # Run rule-based detection first (handles both historical and today logic)
    rule_result = detect_fake_discount(
        df,
        product_code,
        evaluation_date,
        current_price=current_price,
        claimed_original_price=claimed_original_price,
        window_days=window_days,
        drop_threshold=drop_threshold,
        spike_z_threshold=spike_z_threshold,
    )

    # ML anomaly score is product-level (since today isn't in dataset)
    ml_score = float(score_dict.get(product_code, 0.0))

    # Adjust status based on ML score
    final_status = rule_result.get("discount_status", "Unknown")
    if final_status == "Suspicious" and ml_score > 0.5:
        final_status = "Highly Suspicious"
    elif final_status == "Genuine" and ml_score > 0.75:
        final_status = "Highly Suspicious"

    # Update result dict
    rule_result["ml_anomaly_score"] = round(ml_score, 3)
    rule_result["discount_status"] = final_status
    rule_result["explanation"] += f" ML anomaly score: {ml_score:.3f} (higher = more abnormal)."

    return rule_result
