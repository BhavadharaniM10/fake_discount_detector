import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from .config import (
    IFOREST_N_ESTIMATORS,
    IFOREST_CONTAMINATION,
    IFOREST_RANDOM_STATE,
    ENGINEERED_FEATURES_PATH,
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
    **kwargs,
) -> dict:
    """
    Combine rule-based detection with ML anomaly score.
    """
    rule_result = detect_fake_discount(df, product_code, evaluation_date, **kwargs)

    ml_score = float(score_dict.get(product_code, 0.0))

    final_status = rule_result["discount_status"]
    if rule_result["discount_status"] == "Suspicious" and ml_score > 0.5:
        final_status = "Highly Suspicious"

    rule_result["ml_anomaly_score"] = round(ml_score, 3)
    rule_result["final_status"] = final_status
    rule_result["explanation"] += (
        f" ML anomaly score: {ml_score:.3f} (higher = more abnormal)."
    )

    return rule_result
