import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.config import ENGINEERED_FEATURES_PATH
from src.rule_detection import detect_fake_discount, generate_explanation
from src.ml_detection import train_isolation_forest
from src.visualization import plot_detection
from src.utils import setup_logging

logger = setup_logging()

@st.cache_data
def load_engineered_features():
    if not os.path.exists(ENGINEERED_FEATURES_PATH):
        st.error("engineered_features.parquet not found. Run the pipeline first (python -m app.main).")
        st.stop()
    return pd.read_parquet(ENGINEERED_FEATURES_PATH)

@st.cache_resource
def get_ml_scores(engineered_df: pd.DataFrame):
    _, score_dict = train_isolation_forest(engineered_df)
    return score_dict

def main():
    st.title("🛒 Fake Discount Detector")
    st.write("Analyze Amazon product price history to detect potentially fake discounts.")

    engineered_df = load_engineered_features()

    st.sidebar.header("Controls")

    product_codes = sorted(engineered_df["product_code"].unique())
    product_code = st.sidebar.selectbox("Product Code", product_codes)

    product_dates = engineered_df[engineered_df["product_code"] == product_code]["order_date"]
    min_date, max_date = product_dates.min(), product_dates.max()

    eval_date = st.sidebar.date_input(
        "Evaluation Date",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
    )

    use_hybrid = st.sidebar.checkbox("Use ML Hybrid Detection (Isolation Forest)", value=True)

    if st.sidebar.button("Run Detection"):
        st.subheader("🔍 Detection Result")

        if use_hybrid:
            score_dict = get_ml_scores(engineered_df)
            from src.ml_detection import hybrid_detect_fake_discount
            result = hybrid_detect_fake_discount(
                engineered_df,
                score_dict,
                product_code,
                str(eval_date),
            )
        else:
            result = detect_fake_discount(
                engineered_df,
                product_code,
                str(eval_date),
            )

        st.json(result)

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

        # Chart
        st.subheader("📈 Price History Chart")
        fig = plot_detection(engineered_df, result)
        st.pyplot(fig)

if __name__ == "__main__":
    main()
