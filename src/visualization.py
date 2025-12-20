import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from .config import PRICE_PLOT_PATH
from .utils import setup_logging

logger = setup_logging()

def plot_detection(
    df: pd.DataFrame,
    result: dict,
    save_path: str = PRICE_PLOT_PATH,
    show_spikes: bool = True,
):
    """
    Plot price history with evaluation date, peak, drop, and spike markers.
    In today mode, plots the marker above the last dataset date to avoid gaps.
    Saves PNG and returns matplotlib figure.
    """
    product_code = result["product_code"]
    eval_date = pd.to_datetime(result["evaluation_date"]).normalize()
    current_price = result.get("current_price", None)
    claimed_original = result.get("claimed_original_price", None)

    product_df = df[df["product_code"] == product_code].sort_values("order_date")
    if product_df.empty:
        raise ValueError(f"No data for product_code: {product_code}")

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(
        product_df["order_date"],
        product_df["daily_mean_price"],
        label="Daily Mean Price",
        color="blue",
        linewidth=2,
        marker="o",
        markersize=4,
    )

    today = pd.Timestamp.today().normalize()
    last_date = product_df["order_date"].max()

    # -------------------------------
    # Historical mode
    # -------------------------------
    if eval_date in product_df["order_date"].dt.normalize().values:
        ax.axvline(eval_date, color="red", linestyle="--", label="Evaluation Date")

    # -------------------------------
    # Real-time mode (today)
    # -------------------------------
    elif eval_date == today and current_price is not None:
        # Plot today’s marker directly above last dataset date
        ax.scatter(
            last_date,
            current_price,
            color="red",
            s=180,
            marker="o",
            label=f"Today Price (${current_price:.2f})",
        )
        # Annotate clearly that this is today’s user-entered price
        ax.annotate(
            f"Today (user): ${current_price:.2f}",
            (last_date, current_price),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            color="red",
            fontsize=12,
            fontweight="bold",
        )

        # Draw drop line if claimed original >= current
        if claimed_original is not None and claimed_original >= current_price:
            ax.plot(
                [last_date, last_date],
                [claimed_original, current_price],
                color="red",
                linestyle="--",
                linewidth=3,
                label="Claimed Drop",
            )
            ax.scatter(
                last_date,
                claimed_original,
                color="green",
                s=150,
                marker="^",
                label=f"Claimed Original (${claimed_original:.2f})",
            )
            ax.fill_betweenx(
                [current_price, claimed_original],
                last_date,
                last_date,
                color="red",
                alpha=0.15,
                label="Drop Area",
            )

    # -------------------------------
    # Spike markers
    # -------------------------------
    if show_spikes and "rolling_z_score" in product_df.columns:
        spikes = product_df[product_df["rolling_z_score"] > 2.0]
        if not spikes.empty:
            ax.scatter(
                spikes["order_date"],
                spikes["daily_mean_price"],
                color="purple",
                s=120,
                marker="*",
                label="Detected Spikes (z > 2.0)",
            )

    # -------------------------------
    # Final chart formatting
    # -------------------------------
    ax.set_title(
        f"Price History for {product_code}\nStatus: {result.get('final_status', result.get('discount_status'))}"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Mean Price ($)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved detection plot to {save_path}")

    return fig
