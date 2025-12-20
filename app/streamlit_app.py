import os
import sys
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Project imports and paths
# -----------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.config import ENGINEERED_FEATURES_PATH
from src.rule_detection import detect_fake_discount, generate_explanation
from src.ml_detection import train_isolation_forest, hybrid_detect_fake_discount
from src.visualization import plot_detection
from src.utils import setup_logging

logger = setup_logging()

# -----------------------------
# Cached loaders
# -----------------------------
@st.cache_data
def load_engineered_features() -> pd.DataFrame:
    if not os.path.exists(ENGINEERED_FEATURES_PATH):
        st.error("engineered_features.parquet not found. Run the pipeline first (python -m app.main).")
        st.stop()
    df = pd.read_parquet(ENGINEERED_FEATURES_PATH)
    if not pd.api.types.is_datetime64_any_dtype(df["order_date"]):
        df["order_date"] = pd.to_datetime(df["order_date"])
    return df

@st.cache_resource
def get_ml_scores(engineered_df: pd.DataFrame):
    _, score_dict = train_isolation_forest(engineered_df)
    return score_dict

# -----------------------------
# Main app
# -----------------------------
def main():
    st.title("🛒 Fake Discount Detector")
    st.write(
        "Analyze Amazon product price history and detect potentially fake discounts "
        "using historical patterns or user-entered prices for today."
    )

    # Load engineered features
    engineered_df = load_engineered_features()

    # Sidebar controls
    st.sidebar.header("Controls")

    product_codes = sorted(engineered_df["product_code"].unique())
    product_code = st.sidebar.selectbox("Product Code (ASIN)", product_codes)

    product_df = engineered_df[engineered_df["product_code"] == product_code]
    product_dates = product_df["order_date"].dt.normalize()
    min_date, max_date = product_dates.min(), product_dates.max()

    eval_date = st.sidebar.date_input(
        "Evaluation Date",
        value=max_date,
        min_value=min_date,
        max_value=pd.Timestamp.today().date(),
    )
    eval_date = pd.to_datetime(eval_date).normalize()
    today = pd.Timestamp.today().normalize()

    use_hybrid = st.sidebar.checkbox("Use ML Hybrid Detection (Isolation Forest)", value=True)

    # -----------------------------
    # Mode selection
    # -----------------------------
    if eval_date == today:
        # Real-time mode (today only)
        st.warning("Real-time mode: please enter current and claimed original price.")
        current_price = st.number_input("Current Price", min_value=0.01, step=0.01)
        original_price = st.number_input("Claimed Original Price", min_value=0.01, step=0.01)

        if original_price < current_price:
            st.warning("Claimed original price is less than current price. This may indicate no discount or invalid input.")

        is_historical = False
    else:
        # Historical mode (any date, backend will anchor if needed)
        st.info("Historical mode: using dataset price or nearest available date.")
        current_price = None
        original_price = None
        is_historical = True

    # -----------------------------
    # Run Detection
    # -----------------------------
    st.subheader("Step 3: Run Detection")

    if st.sidebar.button("Run Detection"):
        st.subheader("🔍 Detection Result")

        if not is_historical and current_price <= 0:
            st.error("Please provide a valid current price (> 0).")
            return

        if use_hybrid:
            score_dict = get_ml_scores(engineered_df)
            result = hybrid_detect_fake_discount(
                engineered_df,
                score_dict,
                product_code,
                str(eval_date),
                current_price=current_price,
                claimed_original_price=original_price,
            )
        else:
            result = detect_fake_discount(
                engineered_df,
                product_code,
                str(eval_date),
                current_price=current_price,
                claimed_original_price=original_price,
            )

        # Display raw result dict
        st.json(result)

        # Explanation
        explanation = generate_explanation(result)
        st.subheader("📝 Explanation")
        st.write(explanation)

        # Key metrics
        st.subheader("📊 Key Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Drop %", f"{result.get('drop_percentage', 0.0) * 100:.1f}%")
        with col2:
            st.metric("Volatility Score", f"{result.get('volatility_score', 0.0):.2f}")
        with col3:
            if "ml_anomaly_score" in result:
                st.metric("ML Anomaly Score", f"{result['ml_anomaly_score']:.3f}")
            else:
                st.metric("ML Anomaly Score", "N/A")

        # Price history chart
        st.subheader("📈 Price History Chart")
        fig = plot_detection(engineered_df, result)
        st.pyplot(fig)


if __name__ == "__main__":
    main()
