"""
Forecasting Models — Extended Multi-Benchmark
==============================================
Tests:
  - Feature configs: Temporal only, + Weather, + Gust, + Strings, Full
  - Feature configs: Temporal only, + Weather, + Gust, + Strings, Full
  - Architectures: Linear Regression, Gradient Boosting,
                   SARIMA, Prophet, Diverse Ensemble
"""
import os, sys, warnings, joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.feature_engineering import (engineer_features, build_daily_dataset,
                                     add_temporal_features)

warnings.filterwarnings("ignore")

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(BASE, "data", "processed", "cleaned_weather_data.csv")
FIGS = os.path.join(BASE, "reports", "figures")
MODELS_DIR = os.path.join(BASE, "models")
os.makedirs(FIGS, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

TARGET = "temperature_celsius"


def _save(name):
    plt.savefig(os.path.join(FIGS, name), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → saved {name}")


def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def split(df, test_frac=0.2):
    n = int(len(df) * (1 - test_frac))
    return df.iloc[:n], df.iloc[n:]


def eval_model(name, y_true, y_pred):
    return {
        "model": name,
        "MAE": round(mean_absolute_error(y_true, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        "MAPE": round(mape(y_true, y_pred), 2),
        "R2": round(r2_score(y_true, y_pred), 4),
        "Med_AE": round(median_absolute_error(y_true, y_pred), 4),
    }


# ── Feature groups ─────────────────────────────────────────────────────────
TEMPORAL = ["dayofyear", "month", "weekday",
            "lag_1", "lag_2", "lag_3", "lag_7", "lag_14",
            "rolling_7", "rolling_30"]

NUMERIC_WX = ["wind_kph", "pressure_mb", "precip_mm",
              "humidity", "cloud", "visibility_km", "uv_index"]

GUST = ["gust_kph"]

STRINGS = ["wind_dir_sin", "wind_dir_cos", "daylight_hours", "moon_phase_ord",
           "cond_clear", "cond_cloudy", "cond_fog",
           "cond_heavy_rain", "cond_light_rain", "cond_snow", "cond_thunder"]

FEAT_CONFIGS = {
    "A) Temporal only":       TEMPORAL,
    "B) + Weather":           TEMPORAL + NUMERIC_WX,
    "C) + Gust":              TEMPORAL + NUMERIC_WX + GUST,
    "D) + Strings":           TEMPORAL + NUMERIC_WX + STRINGS,
    "E) Full":                TEMPORAL + NUMERIC_WX + GUST + STRINGS,
}


# ═══════════════════════════════════════════════════════════════════════════
# ML MODELS
# ═══════════════════════════════════════════════════════════════════════════
def train_ml_models(train, test, features):
    X_tr, y_tr = train[features].values, train[TARGET].values
    X_te = test[features].values

    results = {}

    # Linear Regression (Linear baseline)
    m = LinearRegression().fit(X_tr, y_tr)
    results["Linear Regression"] = (m.predict(X_te), m)

    # Gradient Boosting (Tree baseline - best performer)
    m = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                  learning_rate=0.05, random_state=42).fit(X_tr, y_tr)
    results["Gradient Boosting"] = (m.predict(X_te), m)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# TIME-SERIES MODELS
# ═══════════════════════════════════════════════════════════════════════════
def run_sarima(train_series, n_test):
    """SARIMA(2,1,1)(1,1,1,7) — weekly seasonality."""
    print("    [SARIMA] fitting (2,1,1)(1,1,1,7) …")
    model = SARIMAX(train_series, order=(2, 1, 1),
                    seasonal_order=(1, 1, 1, 7),
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    fit = model.fit(disp=False)
    pred = fit.forecast(steps=n_test)
    return pred.values


def run_prophet(train_df, test_df):
    print("    [Prophet] fitting …")
    train_p = pd.DataFrame({"ds": train_df.index, "y": train_df[TARGET].values})
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                daily_seasonality=False, changepoint_prior_scale=0.05)
    m.fit(train_p)
    return m.predict(pd.DataFrame({"ds": test_df.index}))["yhat"].values


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  FORECASTING BENCHMARKS")
    print("=" * 70)

    # Load & prepare
    print("\n[load] Reading data …")
    df = pd.read_csv(DATA, parse_dates=["last_updated"])
    df = engineer_features(df)
    daily = build_daily_dataset(df, include_strings=True, include_gust=True)
    daily = add_temporal_features(daily)
    train, test = split(daily)
    y_test = test[TARGET].values
    print(f"[load] Daily: {daily.shape} | Train: {len(train)} | Test: {len(test)}")

    all_results = []

    print("\n━━━ PART 1: Time-Series Models (Univariate) ━━━")
    sarima_pred = run_sarima(train[TARGET], len(test))
    r = eval_model("SARIMA", y_test, sarima_pred)
    r["benchmark"] = "Time-Series"
    all_results.append(r)

    prophet_pred = run_prophet(train, test)
    r = eval_model("Prophet", y_test, prophet_pred)
    r["benchmark"] = "Time-Series"
    all_results.append(r)

    # ── PART 2: Feature config benchmarks & Diverse Ensemble ───────────
    print("\n━━━ PART 2: Feature Configs ━━━")
    for cfg, feats in FEAT_CONFIGS.items():
        avail = [f for f in feats if f in daily.columns]
        print(f"\n  {cfg} ({len(avail)} features)")
        ml = train_ml_models(train, test, avail)
        
        # Add diverse ensemble: GB + LR + Prophet
        ens_pred = (ml["Gradient Boosting"][0] + ml["Linear Regression"][0] + prophet_pred) / 3.0
        ml["Diverse Ensemble (GB+LR+Prophet)"] = (ens_pred, None)

        for mname, (pred, _) in ml.items():
            r = eval_model(mname, y_test, pred)
            r["benchmark"] = cfg
            all_results.append(r)


    # ── Results ────────────────────────────────────────────────────────
    res_df = pd.DataFrame(all_results)
    print("\n" + "=" * 70)
    print("  FULL RESULTS")
    print("=" * 70)
    print(res_df.to_string(index=False))
    res_df.to_csv(os.path.join(BASE, "reports", "benchmark_results.csv"), index=False)

    # Summaries
    ens_name = "Diverse Ensemble (GB+LR+Prophet)"
    ens_fc = res_df[(res_df["model"] == ens_name) &
                    res_df["benchmark"].isin(FEAT_CONFIGS.keys())]
    print("\n--- ENSEMBLE per Feature Config ---")
    print(ens_fc[["benchmark", "MAE", "RMSE", "MAPE", "R2", "Med_AE"]].to_string(index=False))

    best = res_df.loc[res_df["RMSE"].idxmin()]
    print(f"\n✓ Best: {best['model']} @ {best['benchmark']}  RMSE={best['RMSE']}")

    # Save best GB model
    best_feats = [f for f in FEAT_CONFIGS["C) + Gust"] if f in daily.columns]
    ml_c = train_ml_models(train, test, best_feats)
    joblib.dump(ml_c["Gradient Boosting"][1], os.path.join(MODELS_DIR, "best_model.pkl"))
    ml_c["Diverse Ensemble (GB+LR+Prophet)"] = ((ml_c["Gradient Boosting"][0] + ml_c["Linear Regression"][0] + prophet_pred) / 3.0, None)

    # ════════════════════════════════════════════════════════════════════
    # PLOTS
    # ════════════════════════════════════════════════════════════════════

    # 1. Ensemble RMSE per config
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#4682B4", "#5F9EA0", "#2E8B57", "#CD853F", "#B22222"]
    bars = ax.bar(range(len(ens_fc)), ens_fc["RMSE"].values, color=colors)
    ax.set_xticks(range(len(ens_fc)))
    ax.set_xticklabels(ens_fc["benchmark"].values, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("RMSE (°C)")
    ax.set_title("Ensemble RMSE by Feature Configuration")
    ax.bar_label(bars, fmt="%.3f", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save("benchmark_ensemble_rmse.png")

    # 3. All models RMSE ranking
    config_c = res_df[res_df["benchmark"] == "C) + Gust"]
    ts = res_df[res_df["benchmark"] == "Time-Series"]
    compare = pd.concat([config_c, ts], ignore_index=True).sort_values("RMSE")
    fig, ax = plt.subplots(figsize=(10, 5))
    c = ["#2E8B57" if i == 0 else "#4682B4" for i in range(len(compare))]
    bars = ax.barh(range(len(compare)), compare["RMSE"].values, color=c)
    ax.set_yticks(range(len(compare)))
    ax.set_yticklabels(compare["model"].values, fontsize=9)
    ax.set_xlabel("RMSE (°C)")
    ax.set_title("All Models RMSE Ranking")
    ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    plt.tight_layout()
    _save("all_models_rmse.png")

    # 4. Forecast comparison
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(test.index, y_test, label="Actual", linewidth=2.5, color="black")
    for mn, (pred, _) in ml_c.items():
        if mn == "Diverse Ensemble (GB+LR+Prophet)":
            ax.plot(test.index, pred, "--", label=mn, linewidth=2, color="green")
        else:
            ax.plot(test.index, pred, alpha=0.5, linewidth=1, label=mn)
    ax.plot(test.index, sarima_pred, ":", alpha=0.7, linewidth=1.5, label="SARIMA")
    ax.plot(test.index, prophet_pred, ":", alpha=0.7, linewidth=1.5, label="Prophet")
    ax.set_title("Temperature Forecast (°C) — All Models (Climatic Trend)")
    ax.set_ylabel("Mean Temperature (°C)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save("forecast_comparison.png")

    # 5. Residuals (top 3 ML + best TS)
    top_models = ["Linear Regression", "Gradient Boosting", "Diverse Ensemble (GB+LR+Prophet)"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, mn in zip(axes[:3], top_models):
        pred = ml_c[mn][0]
        res = y_test - pred
        ax.hist(res, bins=25, color="#5F9EA0", alpha=0.7, edgecolor="white")
        ax.axvline(0, color="red", linestyle="--", alpha=0.5)
        name_short = mn.replace("Diverse Ensemble (GB+LR+Prophet)", "Diverse Ensemble")
        ax.set_title(name_short, fontsize=9)
        ax.set_xlabel("Residual (°C)")
        ax.annotate(f"μ={res.mean():.2f}\nσ={res.std():.2f}\nR²={r2_score(y_test, pred):.2f}",
                    xy=(0.95, 0.95), xycoords="axes fraction",
                    ha="right", va="top", fontsize=7,
                    bbox=dict(boxstyle="round", facecolor="lightyellow"))
    # SARIMA residuals
    res_s = y_test - sarima_pred
    axes[3].hist(res_s, bins=25, color="#CD853F", alpha=0.7, edgecolor="white")
    axes[3].axvline(0, color="red", linestyle="--", alpha=0.5)
    axes[3].set_title("SARIMA", fontsize=9)
    axes[3].set_xlabel("Residual (°C)")
    axes[3].annotate(f"μ={res_s.mean():.2f}\nσ={res_s.std():.2f}\nR²={r2_score(y_test, sarima_pred):.2f}",
                     xy=(0.95, 0.95), xycoords="axes fraction",
                     ha="right", va="top", fontsize=7,
                     bbox=dict(boxstyle="round", facecolor="lightyellow"))
    plt.suptitle("Residual Distributions", fontsize=12)
    plt.tight_layout()
    _save("residual_distributions.png")

    # 6. Actual vs Predicted (best ML)
    best_ml = config_c.loc[config_c["RMSE"].idxmin(), "model"]
    if best_ml in ml_c:
        bp = ml_c[best_ml][0]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_test, bp, s=20, alpha=0.6, color="#4682B4")
        lims = [min(y_test.min(), bp.min())-1, max(y_test.max(), bp.max())+1]
        ax.plot(lims, lims, "--", color="red", alpha=0.5, label="Perfect")
        ax.set_xlabel("Actual (°C)")
        ax.set_ylabel("Predicted (°C)")
        ax.set_title(f"Actual vs Predicted — {best_ml}")
        ax.legend()
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        _save("actual_vs_predicted.png")

    # 7. Feature importance (GB)
    _, gb_model = ml_c["Gradient Boosting"]
    imp = pd.Series(gb_model.feature_importances_,
                    index=best_feats).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    imp.plot(kind="barh", ax=ax, color="#5F9EA0")
    ax.set_title("Feature Importance — Gradient Boosting")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    _save("feature_importance_gradient_boosting.png")

    print("\n✓ Forecasting complete.\n")


if __name__ == "__main__":
    main()
