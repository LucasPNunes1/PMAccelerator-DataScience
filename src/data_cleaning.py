"""
Data Cleaning & Preprocessing
------------------------------
Handles missing values, sentinel / impossible values,
and exports a cleaned dataset ready for EDA and modelling.
"""
import pandas as pd
import numpy as np
import os

# ── Columns to drop ───────────────────────────────────────────────────────
# 1. Pure unit duplicates (imperial versions of existing metric columns)
# 2. feels_like_celsius: this is a DERIVED variable calculated from
#    temperature_celsius + humidity + wind. Since we already have all three
#    source variables, it adds zero new information — any model can learn
#    that mapping from the inputs.
# 3. gust_mph: imperial duplicate of gust_kph (which we KEEP because
#    gust ≠ wind — their divergence signals turbulence/storms)
COLS_TO_DROP = [
    "temperature_fahrenheit", "feels_like_fahrenheit",
    "feels_like_celsius",  # derived from temp + humidity + wind
    "wind_mph", "pressure_in", "precip_in",
    "visibility_miles", "gust_mph",
]

# ── Sentinel values: ONLY where evidence exists in describe() ──────────────
# Evidence from raw data describe():
#   air_quality_Carbon_Monoxide min = -9999
#   air_quality_Sulphur_dioxide  min = -9999
#   air_quality_PM10             min = -1848
SENTINEL_RULES = {
    "air_quality_Carbon_Monoxide": 0,
    "air_quality_Sulphur_dioxide": 0,
    "air_quality_PM10": 0,
}


def clean_weather_data(input_path: str, output_path: str) -> pd.DataFrame:
    print(f"[clean] Loading raw data from {input_path} …")
    df = pd.read_csv(input_path)
    print(f"[clean] Raw shape: {df.shape}")


    cols = [c for c in COLS_TO_DROP if c in df.columns]
    if cols:
        df.drop(columns=cols, inplace=True)
        print(f"[clean] Dropped {len(cols)} columns: {cols}")


    df.replace(["No moonrise", "No moonset"], np.nan, inplace=True)
    print("[clean] Replaced astronomical placeholders with NaN.")


    for col, lower in SENTINEL_RULES.items():
        if col in df.columns:
            mask = df[col] < lower
            n = mask.sum()
            if n > 0:
                df.loc[mask, col] = np.nan
                print(f"[clean] {col}: {n} sentinel values (< {lower}) → NaN")


    df["last_updated"] = pd.to_datetime(df["last_updated"])
    print("[clean] Parsed 'last_updated' to datetime.")

    # 5. Handle outliers and structural errors
    drop_mask = pd.Series(False, index=df.index)
    for col, limit in [("wind_kph", 300), ("gust_kph", 300)]:
        if col in df.columns:
            drop_mask = drop_mask | (df[col] > limit)
    
    if "pressure_mb" in df.columns:
        drop_mask = drop_mask | (df["pressure_mb"] < 800) | (df["pressure_mb"] > 1100)
        
    n_dropped = drop_mask.sum()
    if n_dropped > 0:
        df = df[~drop_mask].copy()
        print(f"[clean] Dropped {n_dropped} rows containing physically impossible wind/pressure.")


    # Find the 99.5th percentile of non-zero precipitation to avoid capping everything if largely zero
    if "precip_mm" in df.columns:
        non_zero_precip = df.loc[df["precip_mm"] > 0, "precip_mm"]
        if not non_zero_precip.empty:
            p99 = non_zero_precip.quantile(0.995)
            # Make sure the cap is at least a realistic heavy rain volume (e.g. 250mm) so we don't cap normal stormy days
            cap_val = max(250.0, p99)
            cap_mask = df["precip_mm"] > cap_val
            n_capped = cap_mask.sum()
            df.loc[cap_mask, "precip_mm"] = cap_val
            if n_capped > 0:
                print(f"[clean] Capped {n_capped} extreme precipitation values > {cap_val:.1f} mm.")


    for col in ["country", "location_name"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
    print("[clean] Standardized country and location names to lowercase.")


    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[clean] Saved → {output_path}  shape={df.shape}")
    return df


if __name__ == "__main__":
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw  = os.path.join(base, "data", "raw",       "GlobalWeatherRepository.csv")
    out  = os.path.join(base, "data", "processed",  "cleaned_weather_data.csv")
    clean_weather_data(raw, out)
