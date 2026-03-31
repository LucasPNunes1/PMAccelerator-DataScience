"""
Exploratory Data Analysis — Basic + Advanced (Anomaly Detection)
================================================================
Reads cleaned data, produces:
  • Distribution histograms (temperature, precipitation)
  • Monthly global‑mean temperature trend
  • Correlation heat‑map
  • Anomaly detection (IQR) with visualization
All figures are saved to  reports/figures/.
"""
import os, sys, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(BASE, "data", "processed", "cleaned_weather_data.csv")
FIGS = os.path.join(BASE, "reports", "figures")
os.makedirs(FIGS, exist_ok=True)

NUMERIC_WEATHER = [
    "temperature_celsius", "wind_kph", "pressure_mb", "precip_mm",
    "humidity", "cloud", "visibility_km", "uv_index",
]

AIR_QUALITY = [
    "air_quality_Carbon_Monoxide", "air_quality_Ozone",
    "air_quality_Nitrogen_dioxide", "air_quality_Sulphur_dioxide",
    "air_quality_PM2.5", "air_quality_PM10",
]


def _save(name):
    path = os.path.join(FIGS, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → saved {name}")


# ══════════════════════════════════════════════════════════════════════════
#  1. BASIC EDA
# ══════════════════════════════════════════════════════════════════════════

def basic_stats(df):
    print("\n=== BASIC STATS ===")
    print(f"Shape: {df.shape}")
    print(f"Unique countries: {df['country'].nunique()}")
    print(f"Unique locations: {df['location_name'].nunique()}")
    print(f"Date range: {df['last_updated'].min()} → {df['last_updated'].max()}")
    print(df.describe().T.to_string())


def plot_distributions(df):
    """Temperature & Precipitation distributions."""
    print("\n[EDA] Plotting distributions …")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(df["temperature_celsius"].dropna(), bins=50, kde=True,
                 color="salmon", ax=axes[0])
    axes[0].set_title("Temperature Distribution (°C)")
    axes[0].set_xlabel("Temperature (°C)")

    precip = df["precip_mm"].dropna()
    sns.histplot(precip[precip > 0], bins=50, kde=True,
                 color="steelblue", ax=axes[1])
    axes[1].set_title("Precipitation Distribution (mm) — non‑zero only")
    axes[1].set_xlabel("Precipitation (mm)")

    plt.suptitle("Temperature & Precipitation Distributions", fontsize=14, y=1.02)
    plt.tight_layout()
    _save("distributions_temp_precip.png")


def plot_monthly_trend(df):
    """Monthly global‑mean temperature trend."""
    print("[EDA] Plotting monthly trend …")
    monthly = (
        df.set_index("last_updated")
          .resample("ME")["temperature_celsius"]
          .mean()
    )
    fig, ax = plt.subplots(figsize=(12, 4))
    monthly.plot(ax=ax, marker="o", linewidth=2, color="orangered")
    ax.set_title("Monthly Global Mean Temperature (°C)")
    ax.set_ylabel("Mean Temperature (°C)")
    ax.set_xlabel("Month")
    ax.grid(True, alpha=.3)
    plt.tight_layout()
    _save("monthly_mean_temperature.png")


def plot_correlation(df):
    """Correlation heat‑map for key numeric features."""
    print("[EDA] Plotting correlation heat‑map …")
    cols = [c for c in NUMERIC_WEATHER + AIR_QUALITY if c in df.columns]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=.5, ax=ax)
    ax.set_title("Correlation Heat‑map of Weather & Air‑Quality Features")
    plt.tight_layout()
    _save("correlation_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════
#  2. ADVANCED EDA — ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════

def detect_anomalies_iqr(df, columns, k=1.5):
    """
    IQR‑based anomaly detection.
    Returns a DataFrame with boolean mask columns (<col>_anomaly).
    """
    result = pd.DataFrame(index=df.index)
    summary = []
    for col in columns:
        s = df[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - k * iqr, q3 + k * iqr
        mask = (df[col] < lower) | (df[col] > upper)
        result[f"{col}_anomaly"] = mask
        n = mask.sum()
        pct = n / len(df) * 100
        summary.append(dict(feature=col, anomalies=n, pct=round(pct, 2),
                            lower=round(lower, 2), upper=round(upper, 2)))
    summary_df = pd.DataFrame(summary)
    return result, summary_df


def plot_anomalies(df, anomalies_df, columns):
    """Box‑plots with anomalies highlighted."""
    print("[EDA] Plotting anomaly box‑plots …")
    n = len(columns)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(16, 8))
    axes = axes.flatten()
    for i, col in enumerate(columns):
        ax = axes[i]
        bp = ax.boxplot(df[col].dropna(), vert=True, patch_artist=True,
                        boxprops=dict(facecolor="lightblue"))
        mask = anomalies_df[f"{col}_anomaly"]
        anom_vals = df.loc[mask, col].dropna()
        if len(anom_vals):
            ax.scatter(np.ones(len(anom_vals)), anom_vals, c="red",
                       s=8, alpha=.5, zorder=5, label="anomaly")
        ax.set_title(col.replace("_", " "), fontsize=9)
        ax.tick_params(labelbottom=False)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Anomaly Detection (IQR method, k=1.5)", fontsize=13)
    plt.tight_layout()
    _save("anomaly_boxplots.png")


def plot_anomaly_scatter(df, anomalies_df):
    """Scatter of temperature vs humidity, colored by anomaly status."""
    print("[EDA] Plotting anomaly scatter …")
    mask = anomalies_df["temperature_celsius_anomaly"]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df.loc[~mask, "humidity"], df.loc[~mask, "temperature_celsius"],
               s=3, alpha=.15, label="Normal", color="steelblue")
    ax.scatter(df.loc[mask, "humidity"], df.loc[mask, "temperature_celsius"],
               s=10, alpha=.6, label="Anomaly", color="red")
    ax.set_xlabel("Humidity (%)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Temperature Anomalies vs Humidity")
    ax.legend()
    plt.tight_layout()
    _save("anomaly_temp_humidity.png")


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    sns.set_theme(style="whitegrid")
    df = pd.read_csv(DATA, parse_dates=["last_updated"])

    # ── Basic EDA ──────────────────────────────────────────────────
    basic_stats(df)
    plot_distributions(df)
    plot_monthly_trend(df)
    plot_correlation(df)

    # ── Advanced: anomaly detection ────────────────────────────────
    anomaly_cols = NUMERIC_WEATHER
    anom_df, anom_summary = detect_anomalies_iqr(df, anomaly_cols)
    print("\n=== ANOMALY SUMMARY (IQR, k=1.5) ===")
    print(anom_summary.to_string(index=False))
    anom_summary.to_csv(os.path.join(BASE, "reports", "anomaly_summary.csv"), index=False)

    plot_anomalies(df, anom_df, anomaly_cols)
    plot_anomaly_scatter(df, anom_df)

    print("\n✓ EDA complete.\n")


if __name__ == "__main__":
    main()
