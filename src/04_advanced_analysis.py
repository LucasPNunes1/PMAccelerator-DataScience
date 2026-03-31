"""
Advanced Analyses
=================
Covers the 5 advanced requirements from the PM Accelerator assessment:

1. Climate Analysis — regional/hemispheric temperature trends
2. Environmental Impact — air quality ↔ weather correlations
3. Feature Importance — multiple techniques (RF, permutation, SHAP-like)
4. Spatial Analysis — geographical temperature/climate maps
5. Geographical Patterns — country/continent weather comparison
"""
import os, sys, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from scipy import stats

warnings.filterwarnings("ignore")

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(BASE, "data", "processed", "cleaned_weather_data.csv")
FIGS = os.path.join(BASE, "reports", "figures")
os.makedirs(FIGS, exist_ok=True)

sns.set_theme(style="whitegrid")


def _save(name):
    plt.savefig(os.path.join(FIGS, name), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → saved {name}")


def assign_hemisphere(lat):
    if lat >= 0:
        return "Northern"
    return "Southern"


def assign_climate_zone(lat):
    a = abs(lat)
    if a <= 23.5:
        return "Tropical"
    elif a <= 40:
        return "Subtropical"
    elif a <= 55:
        return "Temperate"
    else:
        return "Polar/Subpolar"


def assign_continent(country):
    """Simplified continent mapping based on common countries."""
    mapping = {
        # Africa
        "Algeria": "Africa", "Angola": "Africa", "Benin": "Africa",
        "Botswana": "Africa", "Burkina Faso": "Africa", "Burundi": "Africa",
        "Cameroon": "Africa", "Central African Republic": "Africa",
        "Chad": "Africa", "Comoros": "Africa", "Congo": "Africa",
        "Ivory Coast": "Africa", "Democratic Republic of the Congo": "Africa",
        "Djibouti": "Africa", "Egypt": "Africa", "Equatorial Guinea": "Africa",
        "Eritrea": "Africa", "Eswatini": "Africa",
        "Ethiopia": "Africa", "Gabon": "Africa", "Gambia": "Africa",
        "Ghana": "Africa", "Guinea": "Africa", "Guinea-Bissau": "Africa",
        "Kenya": "Africa", "Lesotho": "Africa", "Liberia": "Africa",
        "Libya": "Africa", "Madagascar": "Africa", "Malawi": "Africa",
        "Mali": "Africa", "Mauritania": "Africa", "Mauritius": "Africa",
        "Morocco": "Africa", "Mozambique": "Africa", "Namibia": "Africa",
        "Niger": "Africa", "Nigeria": "Africa", "Rwanda": "Africa",
        "Sao Tome and Principe": "Africa", "Senegal": "Africa",
        "Seychelles": "Africa", "Sierra Leone": "Africa", "Somalia": "Africa",
        "South Africa": "Africa", "South Sudan": "Africa", "Sudan": "Africa",
        "Tanzania": "Africa", "Togo": "Africa", "Tunisia": "Africa",
        "Uganda": "Africa", "Zambia": "Africa", "Zimbabwe": "Africa",
        "Cape Verde": "Africa", "Reunion": "Africa",
        # Asia
        "Afghanistan": "Asia", "Armenia": "Asia", "Azerbaijan": "Asia",
        "Bahrain": "Asia", "Bangladesh": "Asia", "Bhutan": "Asia",
        "Brunei": "Asia", "Cambodia": "Asia", "China": "Asia",
        "Georgia": "Asia", "India": "Asia", "Indonesia": "Asia",
        "Iran": "Asia", "Iraq": "Asia", "Israel": "Asia",
        "Japan": "Asia", "Jordan": "Asia", "Kazakhstan": "Asia",
        "Kuwait": "Asia", "Kyrgyzstan": "Asia", "Laos": "Asia",
        "Lebanon": "Asia", "Malaysia": "Asia", "Maldives": "Asia",
        "Mongolia": "Asia", "Myanmar": "Asia", "Nepal": "Asia",
        "North Korea": "Asia", "Oman": "Asia", "Pakistan": "Asia",
        "Palestine": "Asia", "Philippines": "Asia", "Qatar": "Asia",
        "Saudi Arabia": "Asia", "Singapore": "Asia", "South Korea": "Asia",
        "Sri Lanka": "Asia", "Syria": "Asia", "Taiwan": "Asia",
        "Tajikistan": "Asia", "Thailand": "Asia", "Timor-Leste": "Asia",
        "Turkey": "Asia", "Turkmenistan": "Asia",
        "United Arab Emirates": "Asia", "Uzbekistan": "Asia",
        "Vietnam": "Asia", "Yemen": "Asia",
        # Europe
        "Albania": "Europe", "Andorra": "Europe", "Austria": "Europe",
        "Belarus": "Europe", "Belgium": "Europe",
        "Bosnia and Herzegovina": "Europe", "Bulgaria": "Europe",
        "Croatia": "Europe", "Cyprus": "Europe", "Czech Republic": "Europe",
        "Czechia": "Europe",
        "Denmark": "Europe", "Estonia": "Europe", "Finland": "Europe",
        "France": "Europe", "Germany": "Europe", "Greece": "Europe",
        "Hungary": "Europe", "Iceland": "Europe", "Ireland": "Europe",
        "Italy": "Europe", "Kosovo": "Europe", "Latvia": "Europe",
        "Liechtenstein": "Europe", "Lithuania": "Europe",
        "Luxembourg": "Europe", "Malta": "Europe", "Moldova": "Europe",
        "Monaco": "Europe", "Montenegro": "Europe", "Netherlands": "Europe",
        "North Macedonia": "Europe", "Norway": "Europe", "Poland": "Europe",
        "Portugal": "Europe", "Romania": "Europe", "Russia": "Europe",
        "San Marino": "Europe", "Serbia": "Europe", "Slovakia": "Europe",
        "Slovenia": "Europe", "Spain": "Europe", "Sweden": "Europe",
        "Switzerland": "Europe", "Ukraine": "Europe",
        "United Kingdom": "Europe", "Vatican City": "Europe",
        # North America
        "Antigua and Barbuda": "N. America", "Bahamas": "N. America",
        "Barbados": "N. America", "Belize": "N. America",
        "Canada": "N. America", "Costa Rica": "N. America",
        "Cuba": "N. America", "Dominica": "N. America",
        "Dominican Republic": "N. America", "El Salvador": "N. America",
        "Grenada": "N. America", "Guatemala": "N. America",
        "Haiti": "N. America", "Honduras": "N. America",
        "Jamaica": "N. America", "Mexico": "N. America",
        "Nicaragua": "N. America", "Panama": "N. America",
        "Saint Kitts and Nevis": "N. America",
        "Saint Lucia": "N. America",
        "Saint Vincent and the Grenadines": "N. America",
        "Trinidad and Tobago": "N. America",
        "United States of America": "N. America",
        "United States": "N. America",
        # South America
        "Argentina": "S. America", "Bolivia": "S. America",
        "Brazil": "S. America", "Chile": "S. America",
        "Colombia": "S. America", "Ecuador": "S. America",
        "Guyana": "S. America", "Paraguay": "S. America",
        "Peru": "S. America", "Suriname": "S. America",
        "Uruguay": "S. America", "Venezuela": "S. America",
        # Oceania
        "Australia": "Oceania", "Fiji": "Oceania",
        "Kiribati": "Oceania", "Marshall Islands": "Oceania",
        "Micronesia": "Oceania", "Nauru": "Oceania",
        "New Zealand": "Oceania", "Palau": "Oceania",
        "Papua New Guinea": "Oceania", "Samoa": "Oceania",
        "Solomon Islands": "Oceania", "Tonga": "Oceania",
        "Tuvalu": "Oceania", "Vanuatu": "Oceania",
    }
    mapping_lower = {k.lower(): v for k, v in mapping.items()}
    return mapping_lower.get(str(country).lower().strip(), "Other")


# ═══════════════════════════════════════════════════════════════════════════
# 1. CLIMATE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
def climate_analysis(df):
    print("\n━━━ 1. CLIMATE ANALYSIS ━━━")

    df["month"] = df["last_updated"].dt.to_period("M")
    df["hemisphere"] = df["latitude"].apply(assign_hemisphere)
    df["climate_zone"] = df["latitude"].apply(assign_climate_zone)

    # 1a. Monthly temperature trend by hemisphere
    hemi_monthly = (df.groupby(["month", "hemisphere"])["temperature_celsius"]
                    .mean().reset_index())
    hemi_monthly["month"] = hemi_monthly["month"].dt.to_timestamp()

    fig, ax = plt.subplots(figsize=(14, 5))
    for hemi, color in [("Northern", "#CD5C5C"), ("Southern", "#4682B4")]:
        sub = hemi_monthly[hemi_monthly["hemisphere"] == hemi]
        ax.plot(sub["month"], sub["temperature_celsius"], marker="o",
                linewidth=2, label=hemi, color=color)
    ax.set_title("Monthly Mean Temperature by Hemisphere")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save("climate_hemisphere_trends.png")

    # 1b. Temperature by climate zone (boxplot)
    zone_order = ["Tropical", "Subtropical", "Temperate", "Polar/Subpolar"]
    fig, ax = plt.subplots(figsize=(10, 5))
    zone_data = [df[df["climate_zone"] == z]["temperature_celsius"].dropna()
                 for z in zone_order]
    bp = ax.boxplot(zone_data, labels=zone_order, patch_artist=True,
                    boxprops=dict(facecolor="lightblue"))
    colors_zone = ["#FF6B6B", "#FFA07A", "#87CEEB", "#B0E0E6"]
    for patch, color in zip(bp["boxes"], colors_zone):
        patch.set_facecolor(color)
    ax.set_title("Temperature Distribution by Climate Zone")
    ax.set_ylabel("Temperature (°C)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save("climate_zone_boxplot.png")

    # 1c. Precipitation trends by hemisphere
    hemi_precip = (df.groupby(["month", "hemisphere"])["precip_mm"]
                   .mean().reset_index())
    hemi_precip["month"] = hemi_precip["month"].dt.to_timestamp()

    fig, ax = plt.subplots(figsize=(14, 5))
    for hemi, color in [("Northern", "#CD5C5C"), ("Southern", "#4682B4")]:
        sub = hemi_precip[hemi_precip["hemisphere"] == hemi]
        ax.plot(sub["month"], sub["precip_mm"], marker="o",
                linewidth=2, label=hemi, color=color)
    ax.set_title("Monthly Mean Precipitation by Hemisphere")
    ax.set_ylabel("Precipitation (mm)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save("climate_hemisphere_precip.png")

    # 1d. Summary stats per zone
    zone_stats = df.groupby("climate_zone").agg(
        mean_temp=("temperature_celsius", "mean"),
        std_temp=("temperature_celsius", "std"),
        mean_precip=("precip_mm", "mean"),
        mean_humidity=("humidity", "mean"),
        n_records=("temperature_celsius", "count"),
    ).round(2)
    print("\nClimate Zone Summary:")
    print(zone_stats.to_string())

    return df


# ═══════════════════════════════════════════════════════════════════════════
# 2. ENVIRONMENTAL IMPACT
# ═══════════════════════════════════════════════════════════════════════════
def environmental_impact(df):
    print("\n━━━ 2. ENVIRONMENTAL IMPACT ━━━")

    aq_cols = ["air_quality_Carbon_Monoxide", "air_quality_Ozone",
               "air_quality_Nitrogen_dioxide", "air_quality_Sulphur_dioxide",
               "air_quality_PM2.5", "air_quality_PM10"]
    wx_cols = ["temperature_celsius", "wind_kph", "humidity",
               "pressure_mb", "visibility_km"]

    # 2a. Correlation heatmap: weather ↔ air quality
    corr_cols = wx_cols + aq_cols
    corr = df[corr_cols].corr()
    cross = corr.loc[wx_cols, aq_cols]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(cross, annot=True, fmt=".2f", cmap="RdYlBu_r",
                center=0, linewidths=0.5, ax=ax, annot_kws={"size": 9})
    ax.set_title("Weather → Air Quality Correlation")
    plt.tight_layout()
    _save("env_weather_airquality_corr.png")

    # 2b. Wind speed vs pollutant dispersion
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (pollutant, color, label) in zip(axes, [
        ("air_quality_Carbon_Monoxide", "#CD853F", "CO"),
        ("air_quality_PM2.5", "#8B4513", "PM2.5"),
        ("air_quality_Ozone", "#4682B4", "Ozone"),
    ]):
        sub = df[[pollutant, "wind_kph"]].dropna()
        sub = sub[(sub["wind_kph"] < 100) & (sub[pollutant] < sub[pollutant].quantile(0.99))]
        ax.scatter(sub["wind_kph"], sub[pollutant], s=2, alpha=0.1, color=color)
        z = np.polyfit(sub["wind_kph"], sub[pollutant], 1)
        p = np.poly1d(z)
        x_range = np.linspace(sub["wind_kph"].min(), sub["wind_kph"].max(), 100)
        ax.plot(x_range, p(x_range), "--", color="red", linewidth=2, alpha=0.7)
        r = sub["wind_kph"].corr(sub[pollutant])
        ax.set_title(f"Wind Speed vs {label} (r={r:.2f})")
        ax.set_xlabel("Wind Speed (kph)")
        ax.set_ylabel(f"{label} concentration")
    plt.suptitle("Does Wind Help Disperse Pollutants?", fontsize=12, y=1.02)
    plt.tight_layout()
    _save("env_wind_dispersion.png")

    # 2c. Temperature vs Ozone (physical: UV → ozone formation)
    fig, ax = plt.subplots(figsize=(8, 5))
    sub = df[["temperature_celsius", "air_quality_Ozone"]].dropna()
    sub = sub[sub["air_quality_Ozone"] < sub["air_quality_Ozone"].quantile(0.99)]
    ax.scatter(sub["temperature_celsius"], sub["air_quality_Ozone"],
               s=2, alpha=0.1, color="#FF8C00")
    z = np.polyfit(sub["temperature_celsius"], sub["air_quality_Ozone"], 2)
    p = np.poly1d(z)
    x_range = np.linspace(sub["temperature_celsius"].min(),
                          sub["temperature_celsius"].max(), 100)
    ax.plot(x_range, p(x_range), "--", color="red", linewidth=2)
    ax.set_title("Temperature vs Ozone Formation")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Ozone concentration")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save("env_temp_ozone.png")

    # 2d. Air quality by climate zone
    melted = df.melt(id_vars=["climate_zone"], value_vars=aq_cols,
                     var_name="pollutant", value_name="concentration")
    melted["pollutant"] = melted["pollutant"].str.replace("air_quality_", "")
    zone_aq = melted.groupby(["climate_zone", "pollutant"])["concentration"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(12, 5))
    zone_order = ["Tropical", "Subtropical", "Temperate", "Polar/Subpolar"]
    pivot = zone_aq.pivot(index="pollutant", columns="climate_zone",
                          values="concentration")[zone_order]
    pivot.plot(kind="bar", ax=ax, width=0.7)
    ax.set_title("Mean Pollutant Concentration by Climate Zone")
    ax.set_ylabel("Concentration")
    ax.set_xlabel("Pollutant")
    ax.legend(title="Zone")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    _save("env_airquality_by_zone.png")

    for aq in aq_cols:
        short = aq.replace("air_quality_", "")
        for wx in wx_cols:
            r = corr.loc[wx, aq]
            if abs(r) > 0.1:
                print(f"  {wx:<25} ↔ {short:<20} r={r:+.2f}")


# ═══════════════════════════════════════════════════════════════════════════
# 3. FEATURE IMPORTANCE (multiple techniques)
# ═══════════════════════════════════════════════════════════════════════════
def feature_importance_analysis(df):
    print("\n━━━ 3. FEATURE IMPORTANCE (3 techniques) ━━━")

    features = ["wind_kph", "gust_kph", "pressure_mb", "precip_mm",
                "humidity", "cloud", "visibility_km", "uv_index",
                "latitude", "longitude",
                "air_quality_Carbon_Monoxide", "air_quality_Ozone",
                "air_quality_PM2.5"]
    target = "temperature_celsius"

    sub = df[features + [target]].dropna()
    X, y = sub[features], sub[target]

    # 3a. Correlation-based importance
    corr_imp = X.corrwith(y).abs().sort_values(ascending=False)

    # 3b. Gradient Boosting importance (impurity-based)
    gb = GradientBoostingRegressor(n_estimators=100, max_depth=4,
                                   learning_rate=0.1, random_state=42)
    gb.fit(X, y)
    gb_imp = pd.Series(gb.feature_importances_, index=features).sort_values(ascending=False)

    # 3c. Permutation importance (model-agnostic)
    perm = permutation_importance(gb, X, y, n_repeats=10, random_state=42, n_jobs=2)
    perm_imp = pd.Series(perm.importances_mean, index=features).sort_values(ascending=False)

    comparison = pd.DataFrame({
        "|r| (correlation)": corr_imp,
        "GB (impurity)": gb_imp,
        "Permutation": perm_imp,
    })
    comparison["Avg Rank"] = comparison.rank(ascending=False).mean(axis=1)
    comparison = comparison.sort_values("Avg Rank")

    print("\nFeature Importance Comparison:")
    print(comparison.round(4).to_string())

    # Plot: side-by-side comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    titles = ["|r| Correlation", "GB Impurity-based", "Permutation Importance"]
    datasets = [corr_imp, gb_imp, perm_imp]
    colors_list = ["#4682B4", "#2E8B57", "#CD853F"]

    for ax, title, data, color in zip(axes, titles, datasets, colors_list):
        sorted_data = data.sort_values(ascending=True)
        sorted_data.plot(kind="barh", ax=ax, color=color)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Importance")

    plt.suptitle("Feature Importance — 3 Techniques Compared", fontsize=13, y=1.02)
    plt.tight_layout()
    _save("feature_importance_3methods.png")

    # Plot: average rank (consensus)
    avg_rank = comparison["Avg Rank"].sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors_rank = ["#2E8B57" if i < 3 else "#4682B4" for i in range(len(avg_rank))]
    avg_rank.plot(kind="barh", ax=ax, color=colors_rank)
    ax.set_title("Feature Importance — Consensus Ranking (Avg Rank)")
    ax.set_xlabel("Average Rank (lower = more important)")
    plt.tight_layout()
    _save("feature_importance_consensus.png")


# ═══════════════════════════════════════════════════════════════════════════
# 4. SPATIAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
def spatial_analysis(df):
    print("\n━━━ 4. SPATIAL ANALYSIS ━━━")

    # Aggregate per unique coordinate (latitude, longitude)
    # This prevents duplicate string names from merging distinct geographic stations
    loc = df.groupby(["latitude", "longitude"]).agg(
        location_name=("location_name", "first"),
        country=("country", "first"),
        temp=("temperature_celsius", "mean"),
        precip=("precip_mm", "mean"),
        humidity=("humidity", "mean"),
        wind=("wind_kph", "mean"),
        pm25=("air_quality_PM2.5", "mean"),
        uv=("uv_index", "mean"),
    ).reset_index()
    print(f"Total unique geographic points evaluated: {len(loc)}")

    # 4a. Global temperature map
    fig, ax = plt.subplots(figsize=(16, 8))
    scatter = ax.scatter(loc["longitude"], loc["latitude"], c=loc["temp"],
                         cmap="RdYlBu_r", s=18, alpha=0.8, edgecolors="black", linewidths=0.2)
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Mean Temperature (°C)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Global Temperature Distribution — {len(loc)} Unique Weather Stations")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.3, label="Equator")
    ax.axhline(23.5, color="gray", linestyle=":", alpha=0.2)
    ax.axhline(-23.5, color="gray", linestyle=":", alpha=0.2)
    ax.legend(fontsize=8)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 70)
    ax.grid(True, alpha=0.1)
    plt.tight_layout()
    _save("spatial_temperature_map.png")

    # 4b. Global PM2.5 map
    loc_pm = loc.dropna(subset=["pm25"])
    fig, ax = plt.subplots(figsize=(16, 8))
    scatter = ax.scatter(loc_pm["longitude"], loc_pm["latitude"], c=loc_pm["pm25"],
                         cmap="YlOrRd", s=18, alpha=0.8, edgecolors="black", linewidths=0.2,
                         vmax=loc_pm["pm25"].quantile(0.95))
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Mean PM2.5")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Global Air Quality (PM2.5) Distribution — {len(loc_pm)} Unique Weather Stations")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 70)
    ax.grid(True, alpha=0.1)
    plt.tight_layout()
    _save("spatial_pm25_map.png")

    # 4c. Global UV Index map
    loc_uv = loc.dropna(subset=["uv"])
    fig, ax = plt.subplots(figsize=(16, 8))
    scatter = ax.scatter(loc_uv["longitude"], loc_uv["latitude"], c=loc_uv["uv"],
                         cmap="plasma", s=18, alpha=0.8, edgecolors="black", linewidths=0.2)
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Mean UV Index")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Global UV Index Exposure — {len(loc_uv)} Unique Weather Stations")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 70)
    ax.grid(True, alpha=0.1)
    plt.tight_layout()
    _save("spatial_uv_map.png")

    # 4c. Latitude bands analysis
    df["lat_band"] = pd.cut(df["latitude"],
                            bins=[-90, -40, -23.5, 0, 23.5, 40, 90],
                            labels=["<-40°", "-40° to -23.5°",
                                    "-23.5° to 0°", "0° to 23.5°",
                                    "23.5° to 40°", ">40°"])
    band_stats = df.groupby("lat_band", observed=False).agg(
        mean_temp=("temperature_celsius", "mean"),
        mean_precip=("precip_mm", "mean"),
        mean_humidity=("humidity", "mean"),
    ).round(2)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (col, title, color) in zip(axes, [
        ("mean_temp", "Mean Temperature (°C)", "#CD5C5C"),
        ("mean_precip", "Mean Precipitation (mm)", "#4682B4"),
        ("mean_humidity", "Mean Humidity (%)", "#5F9EA0"),
    ]):
        band_stats[col].plot(kind="bar", ax=ax, color=color)
        ax.set_title(title)
        ax.set_xlabel("Latitude Band")
        ax.set_ylabel(title.split("(")[0].strip())
        ax.tick_params(axis="x", rotation=30)
    plt.suptitle("Weather by Latitude Band", fontsize=13, y=1.02)
    plt.tight_layout()
    _save("spatial_latitude_bands.png")


# ═══════════════════════════════════════════════════════════════════════════
# 5. GEOGRAPHICAL PATTERNS
# ═══════════════════════════════════════════════════════════════════════════
def geographical_patterns(df):
    print("\n━━━ 5. GEOGRAPHICAL PATTERNS ━━━")

    # 5a. Temperature by continent
    df["continent"] = df["country"].apply(assign_continent)
    cont_order = ["Africa", "Asia", "Europe", "N. America",
                  "S. America", "Oceania"]
    cont_data_t = [df[df["continent"] == c]["temperature_celsius"].dropna()
                   for c in cont_order]

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(cont_data_t, labels=cont_order, patch_artist=True)
    cont_colors = ["#E6B800", "#FF6B6B", "#4682B4", "#2E8B57", "#9370DB", "#20B2AA"]
    for patch, color in zip(bp["boxes"], cont_colors):
        patch.set_facecolor(color)
    ax.set_title("Temperature Distribution by Continent")
    ax.set_ylabel("Temperature (°C)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save("geo_temp_by_continent.png")

    # 5b. Air quality by continent
    cont_aq = df.groupby("continent").agg(
        CO=("air_quality_Carbon_Monoxide", "mean"),
        Ozone=("air_quality_Ozone", "mean"),
        NO2=("air_quality_Nitrogen_dioxide", "mean"),
        PM25=("air_quality_PM2.5", "mean"),
    ).round(2)
    cont_aq = cont_aq.loc[cont_aq.index.isin(cont_order)]

    fig, ax = plt.subplots(figsize=(10, 5))
    cont_aq.plot(kind="bar", ax=ax, width=0.7)
    ax.set_title("Mean Pollutant Levels by Continent")
    ax.set_ylabel("Concentration")
    ax.set_xlabel("Continent")
    ax.legend(fontsize=8)
    plt.xticks(rotation=0)
    plt.tight_layout()
    _save("geo_airquality_by_continent.png")

    # 5c. Monthly temperature trends by continent
    df["year_month"] = df["last_updated"].dt.to_period("M")
    cont_monthly = (df.groupby(["year_month", "continent"])["temperature_celsius"]
                    .mean().reset_index())
    cont_monthly["year_month"] = cont_monthly["year_month"].dt.to_timestamp()

    fig, ax = plt.subplots(figsize=(14, 6))
    for cont, color in zip(cont_order, cont_colors):
        sub = cont_monthly[cont_monthly["continent"] == cont]
        ax.plot(sub["year_month"], sub["temperature_celsius"],
                marker="o", linewidth=1.5, label=cont, alpha=0.8, color=color)
    ax.set_title("Monthly Temperature Trends by Continent")
    ax.set_ylabel("Mean Temperature (°C)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save("geo_continent_trends.png")

    # 5d. Top 10 hottest and coldest cities
    city_temp = df.groupby(["location_name", "country"])["temperature_celsius"].mean()
    hottest = city_temp.nlargest(10)
    coldest = city_temp.nsmallest(10)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    hottest.sort_values().plot(kind="barh", ax=axes[0], color="#CD5C5C")
    axes[0].set_title("Top 10 Hottest Cities (Mean Temp)")
    axes[0].set_xlabel("Mean Temperature (°C)")

    coldest.sort_values(ascending=False).plot(kind="barh", ax=axes[1], color="#4682B4")
    axes[1].set_title("Top 10 Coldest Cities (Mean Temp)")
    axes[1].set_xlabel("Mean Temperature (°C)")

    plt.suptitle("Extreme Cities", fontsize=13, y=1.02)
    plt.tight_layout()
    _save("geo_extreme_cities.png")

    print("\nContinent Summary:")
    cont_summary = df.groupby("continent").agg(
        countries=("country", "nunique"),
        cities=("location_name", "nunique"),
        mean_temp=("temperature_celsius", "mean"),
        mean_precip=("precip_mm", "mean"),
        mean_wind=("wind_kph", "mean"),
    ).round(2)
    cont_summary = cont_summary.loc[cont_summary.index.isin(cont_order)]
    print(cont_summary.to_string())


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  ADVANCED ANALYSES")
    print("=" * 70)

    df = pd.read_csv(DATA, parse_dates=["last_updated"])
    print(f"[load] Shape: {df.shape}")

    df = climate_analysis(df)
    environmental_impact(df)
    feature_importance_analysis(df)
    spatial_analysis(df)
    geographical_patterns(df)

    print("\n✓ All advanced analyses complete.\n")


if __name__ == "__main__":
    main()
