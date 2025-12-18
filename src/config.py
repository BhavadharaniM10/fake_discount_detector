import os

# Base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")

RAW_CSV_PATH = os.path.join(DATA_DIR, "amazon-purchases.csv")
CLEANED_CSV_PATH = os.path.join(DATA_DIR, "cleaned_transactions.csv")
DAILY_PRICES_PATH = os.path.join(DATA_DIR, "daily_prices.parquet")
ENGINEERED_FEATURES_PATH = os.path.join(DATA_DIR, "engineered_features.parquet")
PRICE_PLOT_PATH = os.path.join(DATA_DIR, "price_peak_drop_plot.png")

# Time-series and detection defaults
MIN_HISTORY_DAYS = 30          # was 60
SHORT_WINDOW = 7
LONG_WINDOW = 30
RECENT_WINDOW_DAYS = 60        # was 90
DROP_THRESHOLD = 0.18
SPIKE_Z_THRESHOLD = 1.2

# ML (Isolation Forest) defaults
IFOREST_N_ESTIMATORS = 100
IFOREST_CONTAMINATION = 0.1
IFOREST_RANDOM_STATE = 42
