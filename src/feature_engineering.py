"""
Feature Engineering
===================
Transforms string/categorical columns into numeric features
and prepares aggregated daily datasets for forecasting benchmarks.
"""
import pandas as pd
import numpy as np
import os

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# ── Condition text: standardize + group into categories ────────────────────
CONDITION_MAP = {
    # Clear / Sunny
    "sunny": "clear",
    "clear": "clear",
    # Cloudy
    "partly cloudy": "cloudy",
    "cloudy": "cloudy",
    "overcast": "cloudy",
    # Fog / Mist
    "mist": "fog",
    "fog": "fog",
    "freezing fog": "fog",
    # Rain (light)
    "light drizzle": "light_rain",
    "light rain": "light_rain",
    "light rain shower": "light_rain",
    "light freezing rain": "light_rain",
    "patchy light drizzle": "light_rain",
    "patchy light rain": "light_rain",
    "patchy light rain in area with thunder": "light_rain",
    "patchy light rain with thunder": "light_rain",
    "patchy rain nearby": "light_rain",
    "patchy rain possible": "light_rain",
    "freezing drizzle": "light_rain",
    "heavy freezing drizzle": "light_rain",
    # Rain (moderate/heavy)
    "moderate rain": "heavy_rain",
    "moderate rain at times": "heavy_rain",
    "heavy rain": "heavy_rain",
    "heavy rain at times": "heavy_rain",
    "moderate or heavy freezing rain": "heavy_rain",
    "moderate or heavy rain shower": "heavy_rain",
    "moderate or heavy rain in area with thunder": "heavy_rain",
    "moderate or heavy rain with thunder": "heavy_rain",
    "torrential rain shower": "heavy_rain",
    # Snow
    "light snow": "snow",
    "light snow showers": "snow",
    "light sleet": "snow",
    "light sleet showers": "snow",
    "moderate snow": "snow",
    "heavy snow": "snow",
    "patchy light snow": "snow",
    "patchy light snow in area with thunder": "snow",
    "patchy moderate snow": "snow",
    "patchy heavy snow": "snow",
    "patchy snow nearby": "snow",
    "patchy snow possible": "snow",
    "moderate or heavy sleet": "snow",
    "moderate or heavy snow showers": "snow",
    "moderate or heavy snow in area with thunder": "snow",
    "blowing snow": "snow",
    "blizzard": "snow",
    # Thunder
    "thundery outbreaks in nearby": "thunder",
    "thundery outbreaks possible": "thunder",
}

# ── Wind direction: compass → degrees ─────────────────────────────────────
WIND_DIR_DEGREES = {
    "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
    "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
    "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
    "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
}

# ── Moon phase: ordinal encoding ───────────────────────────────────────────
MOON_PHASE_MAP = {
    "New Moon": 0,
    "Waxing Crescent": 1,
    "First Quarter": 2,
    "Waxing Gibbous": 3,
    "Full Moon": 4,
    "Waning Gibbous": 5,
    "Last Quarter": 6,
    "Waning Crescent": 7,
}


def _parse_time_to_hours(time_str):
    """Convert '05:30 AM' → 5.5 (hours since midnight)."""
    try:
        t = pd.to_datetime(time_str, format="%I:%M %p")
        return t.hour + t.minute / 60
    except Exception:
        return np.nan


def engineer_features(df):
    """
    Add encoded features to the DataFrame (in-place).
    Returns the modified DataFrame.
    """
    df = df.copy()

    # 1. condition_text → standardize + map to category
    df["condition_clean"] = (
        df["condition_text"]
        .str.strip()
        .str.lower()
        .map(CONDITION_MAP)
        .fillna("other")
    )

    # 2. wind_direction → circular encoding (sin/cos)
    wind_deg = df["wind_direction"].map(WIND_DIR_DEGREES)
    wind_rad = np.deg2rad(wind_deg)
    df["wind_dir_sin"] = np.sin(wind_rad)
    df["wind_dir_cos"] = np.cos(wind_rad)

    # 3. moon_phase → ordinal
    df["moon_phase_ord"] = df["moon_phase"].map(MOON_PHASE_MAP)

    # 4. sunrise/sunset → daylight hours
    sunrise_h = df["sunrise"].apply(_parse_time_to_hours)
    sunset_h = df["sunset"].apply(_parse_time_to_hours)
    df["daylight_hours"] = sunset_h - sunrise_h
    df.loc[df["daylight_hours"] < 0, "daylight_hours"] = np.nan

    print(f"[feat] Engineered features added. Shape: {df.shape}")
    print(f"[feat] condition_clean categories: {sorted(df['condition_clean'].unique())}")
    return df


def build_daily_dataset(df, include_strings=True, include_gust=True):
    """
    Aggregate per-record data to daily global dataset.
    Returns a DataFrame indexed by date with aggregated features.
    """
    df = df.set_index("last_updated")

    # --- Numeric aggregations (always) ---
    agg = {
        "temperature_celsius": "mean",
        "wind_kph": "mean",
        "pressure_mb": "mean",
        "precip_mm": "sum",
        "humidity": "mean",
        "cloud": "mean",
        "visibility_km": "mean",
        "uv_index": "mean",
    }

    if include_gust and "gust_kph" in df.columns:
        agg["gust_kph"] = "mean"

    # Circular wind direction: aggregate sin/cos then derive angle
    if "wind_dir_sin" in df.columns:
        agg["wind_dir_sin"] = "mean"
        agg["wind_dir_cos"] = "mean"

    # Daylight hours
    if "daylight_hours" in df.columns:
        agg["daylight_hours"] = "mean"

    # Moon
    if "moon_phase_ord" in df.columns:
        agg["moon_phase_ord"] = lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan

    daily = df.resample("D").agg(agg).dropna(subset=["temperature_celsius"])

    # --- String-derived: % of each condition per day ---
    if include_strings and "condition_clean" in df.columns:
        cond_dummies = pd.get_dummies(df["condition_clean"], prefix="cond")
        cond_daily = cond_dummies.resample("D").mean()  # proportion per day
        daily = daily.join(cond_daily)

    daily = daily.dropna()
    return daily


def add_temporal_features(daily, target="temperature_celsius"):
    """Add lag, rolling, and calendar features."""
    daily = daily.copy()
    daily["dayofyear"] = daily.index.dayofyear
    daily["month"] = daily.index.month
    daily["weekday"] = daily.index.weekday
    for lag in [1, 2, 3, 7, 14]:
        daily[f"lag_{lag}"] = daily[target].shift(lag)
    daily["rolling_7"] = daily[target].shift(1).rolling(7).mean()
    daily["rolling_30"] = daily[target].shift(1).rolling(30).mean()
    return daily.dropna()


if __name__ == "__main__":
    data_path = os.path.join(BASE, "data", "processed", "cleaned_weather_data.csv")
    df = pd.read_csv(data_path, parse_dates=["last_updated"])
    df = engineer_features(df)
    daily = build_daily_dataset(df, include_strings=True, include_gust=True)
    daily = add_temporal_features(daily)
    print(f"\nFinal daily dataset: {daily.shape}")
    print(f"Columns: {list(daily.columns)}")
