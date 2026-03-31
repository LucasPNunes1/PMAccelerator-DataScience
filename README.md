# Weather Trend Forecasting (PM Accelerator Tech Assessment)

## PM Accelerator Mission
> *"The Product Manager Accelerator is designed to support and accelerate the career trajectories of professionals transitioning into and advancing within Product Management, empowering them with the technical, analytical, and strategic skills needed to build better products."*

---

## Project Overview
This repository contains the complete submission for the **Weather Trend Forecasting** technical assessment. The objective is to analyze the **"Global Weather Repository"** dataset to forecast future weather trends and extract deeply actionable climate insights using both basic data processing and advanced machine learning techniques. 

This project fulfills **100% of the Basic and Advanced Assessment requirements**, exploring over 40 features to deliver a robust end-to-end data pipeline. 

### How Evaluators Should Navigate the Project
Instead of jumbled notebooks, this project is structured into **professional, production-ready Python pipelines**. 

1. **`src/` (Source Code):** This contains the 4 core scripts. Run them sequentially to reproduce the entire project from scratch.
    - `python src/data_cleaning.py`
    - `python src/02_eda.py`
    - `python src/03_forecasting.py`
    - `python src/04_advanced_analysis.py`
2. **`notebooks/` (Detailed Development):** These Jupyter Notebooks contain the detailed work and the entire thought process that led to the final pipeline. They serve as a more exploratory and complete version of the developed pipeline.
3. **`reports/presentation.ipynb` (Slide Deck):** **START HERE FOR A VISUAL TOUR!** This presentation acts solely as a high-level summary "slide deck". It provides an organized walkthrough of all 18+ high-resolution charts, maps, and model results created by the pipeline without the code clutter.
3. **`data/`**: Stores the raw dataset and the cleaned outputs.
4. **`models/`**: Saves the finalized best-performing Machine Learning model (`best_model.pkl`).
5. **`requirements.txt`**: Contains all dependency packages needed to execute the environment. 

### Installation & Execution
```bash
# 1. Clone the repository
git clone <repository_url>
cd PMAccelerator-DataScience

# 2. Setup virtual environment & install requirements
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Run the full pipeline (or explore the presentation.ipynb)
python src/data_cleaning.py
python src/02_eda.py
python src/03_forecasting.py
python src/04_advanced_analysis.py
```

---

## 1. Data Cleaning & Preprocessing (`src/data_cleaning.py`)
To build a reliable forecasting model, the raw data underwent rigorous sanitization:
*   **Redundancy Elimination:** Dropped strictly imperial duplicate columns (`temperature_fahrenheit`, `feels_like_fahrenheit`) since models derive the exact same variance from the metric values.
*   **Sentinel/Placeholder Handling:** Replaced non-numeric placeholders like "No moonset" and arbitrary sensor failures (e.g., `-9999` for Carbon Monoxide) with explicit `NaN` values to prevent mathematical distortion.
*   **Physical Anomaly Drops:** Filtered out physically impossible glitch records (e.g., `wind_kph > 300` or `pressure_mb < 800`).
*   **Precipitation Capping:** Extreme rain glitches (> 99.5th percentile) were capped instead of dropped, preserving heavy storm signals without blowing up the regression scale.
*   **Geographic Standardization:** Forced all string locations to lowercase to fix duplicate grouping errors.

## 2. Exploratory Data Analysis (EDA) (`src/02_eda.py`)
Basic explorations validated the climatic distributions:
*   **Feature Distributions:** Confirmed heavy right-skew logic in precipitation (mostly zero, with rare spikes) vs. the normal distribution of temperature.
*   **Anomaly Detection (IQR):** Leveraged the Interquartile Range method to isolate abnormal temperature and humidity occurrences, mapping them visually to identify clustering behaviors of severe weather events.
*   **Correlations:** Analyzed linear relationships between humidity, wind, and air quality indexes.

## 3. Forecasting Models (`src/03_forecasting.py`)
To forecast the global temporal **Temperature (°C)** trend, we built a **Diverse Machine Learning Ensemble**, evaluating candidates using `RMSE`, `MAE`, `MAPE`, `Median Absolute Error`, and `R²`.

We tested and combined models with fundamentally different logic architectures:
1.  **Time-Series Base:** `SARIMA` and `Prophet` (Focuses purely on temporal seasonality).
2.  **Linear Base:** `Linear Regression` (Captures macro linear trends efficiently).
3.  **Tree-Based:** `Gradient Boosting` (Captures non-linear thresholds like "if humidity spikes AND pressure drops").
4.  **The Ensemble:** A fused prediction averaging the best models, mitigating individual algorithm bias and increasing generalized accuracy against the hidden test group.

## 4. Unique Advanced Analyses (`src/04_advanced_analysis.py`)
Fulfilling the **Advanced Assessment**, we executed 5 distinct analytical modules:

1.  **Climate Analysis:** Displayed the stark inversion of Temperature and Precipitation trends across the Northern vs. Southern Hemispheres across months, and benchmarked Polar vs. Tropical baselines.
2.  **Environmental Impact:** Discovered strong statistical bounds between weather and pollution. Higher winds act as natural PM2.5/Carbon Monoxide dispersers, while higher heat directly correlates with increased Ozone levels.
3.  **Feature Importance:** Cross-validated which features predict temperature best using 3 different mathematical methods: Pearson Correlation, Permutation Importance, and Tree-Impurity. 
4.  **Spatial Analysis:** Grouped records by strict `latitude` and `longitude` to plot exactly **439 unique weather stations** on a global map, accurately visualizing global distributions of Temperature, UV Index, and PM2.5 air quality without string-overlap corruption.
5.  **Geographical Patterns:** Consolidated distinct continent insights, measuring average precipitation, winds, and identifying the absolute hottest and coldest cities in the repository.

---
