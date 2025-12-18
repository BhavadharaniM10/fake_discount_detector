import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.data_cleaning import load_and_clean_data
from src.aggregation import aggregate_time_series
from src.feature_engineering import engineer_features
from src.ml_detection import load_engineered_features, train_isolation_forest
from src.utils import setup_logging

logger = setup_logging()

def main():
    logger.info("Starting full pipeline...")

    # 1. Clean data
    clean_df = load_and_clean_data()

    # 2. Aggregate time series
    aggregated_df = aggregate_time_series()

    # 3. Feature engineering
    engineered_df = engineer_features()

    # 4. Train Isolation Forest (ML anomaly)
    _, score_dict = train_isolation_forest(engineered_df)

    logger.info("Pipeline finished successfully.")
    return clean_df, aggregated_df, engineered_df, score_dict

if __name__ == "__main__":
    main()
