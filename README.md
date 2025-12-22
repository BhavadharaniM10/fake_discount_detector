# 🛒 Fake-Discount-Detector

This project analyzes Amazon product price histories to detect potentially fake discounts using a combination of **rule-based logic** and **machine learning (Isolation Forest)**. It supports both **historical evaluation** and **real-time user-entered price analysis**, with transparent explanations and annotated charts.

---

## ✨ Features
- Detects fake discounts using historical price patterns with **anchoring to the closest previous available date** if the exact date is missing.
- Supports real-time mode with user-entered current and claimed original prices.
- Computes key metrics: **drop %**, **volatility score**, **ML anomaly score**.
- Annotated price history chart with spikes, drop lines, and user markers.
- Streamlit UI for interactive analysis.

---

## 🧩 Detection Logic

### Historical Mode
Used when the evaluation date is in the dataset or anchored to the closest previous available date.

**Conditions for Detection:**
- At least 5 records in the recent window (default: last 90 days).
- Claimed original price ≥ current price.

**Detection Outcomes:**

| Status        | Conditions                                                                 |
|---------------|----------------------------------------------------------------------------|
| Genuine       | Drop % < threshold (e.g. 20%), no spike, low volatility                    |
| Suspicious    | Drop % ≥ threshold, spike detected, high volatility                        |
| No Discount   | Claimed original < current price                                           |
| Insufficient Data | Fewer than 5 records in recent window                                  |

---

### Real-Time Mode (Today)
Used when user enters today’s price manually.

**Conditions for Detection:**
- At least 3 records in the recent window (relaxed threshold).
- Claimed original price ≥ current price.

**Detection Outcomes:**

| Status        | Conditions                                                                 |
|---------------|----------------------------------------------------------------------------|
| Genuine       | Drop % < threshold, no spike, low volatility                               |
| Suspicious    | Drop % ≥ threshold, spike detected, high volatility                        |
| No Discount   | Claimed original < current price                                           |
| Limited Data  | Fewer than 3 records in recent window                                      |

**Additional Notes:**
- Today’s price is plotted above the last dataset date to avoid chart gaps.
- Explanation includes anchoring note: *“Dataset ends on YYYY-MM-DD. Today’s evaluation is anchored to that date, so results are approximate.”*

---

## 📊 Key Metrics
- **Drop %**: `(claimed_original - current_price) / claimed_original`
- **Volatility Score**: Normalized coefficient of variation in recent window.
- **ML Anomaly Score**: Isolation Forest score (higher = more abnormal).
- **Spike Detection**: Rolling z-score > configured threshold (default: 2.0).

---

## 📈 Chart Annotations
- **Blue line**: Daily Mean Price  
- **Red dot**: Today’s Price (user-entered)  
- **Green triangle**: Claimed Original Price  
- **Red dashed line**: Claimed Drop  
- **Pink shaded area**: Drop Region  
- **Purple star**: Detected Spike  

---

## 🧪 Testing Tips
To test today mode:
- Use products with ≥ 3 records in their last 90-day window.
- Use realistic claimed original and current prices.
- Check chart for spike markers and drop annotations.

---

## ⚙️ Configuration Parameters
Defined in `config.py`:
- **DROP_THRESHOLD**: % drop to flag as suspicious (default: 20%)
- **SPIKE_Z_THRESHOLD**: z-score threshold for spike detection (default: 2.0)
- **RECENT_WINDOW_DAYS**: window size for recent history (default: 90)

---

## 🏁 Motto
**"Transparency First: Every discount explained, every spike exposed."**
