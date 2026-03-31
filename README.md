# Weather Trend Forecasting (PM Accelerator Tech Assessment)

## PM Accelerator Mission
> *"The Product Manager Accelerator Program is designed to support PM professionals through every stage of their careers. From students looking for entry-level jobs to Directors looking to take on a leadership role, our program has helped over hundreds of students fulfill their career aspirations."*

---

## Project Overview
This repository contains a data science pipeline designed to analyze and forecast global weather patterns using the **Global Weather Repository** dataset. The project implements a complete end-to-end workflow, from data sanitization and exploratory analysis to predictive modeling and geospatial visualization.

### Repository Structure
The project is organized into modular Python scripts and Jupyter Notebooks to ensure reproducibility and clarity:

1.  **`src/` (Source Code):** Modular scripts for automated execution of the pipeline.
    *   `data_cleaning.py`: Sanitization and feature engineering.
    *   `02_eda.py`: Statistical distributions and anomaly detection.
    *   `03_forecasting.py`: Time-series and regression model training.
    *   `04_advanced_analysis.py`: Geospatial and environmental correlation modules.
2.  **`notebooks/`:** Detailed research and development history, containing the exploratory logic behind each pipeline step.
3.  **`reports/presentation.ipynb`:** A consolidated walkthrough of the project's main findings, including 25+ visualizations and model performance benchmarks.
4.  **`models/`**: Storage for the serialized best-performing model (`best_model.pkl`).
5.  **`data/`**: Directories for raw and processed datasets.

### Installation & Setup
```bash
# 1. Clone the repository
git clone <repository_url>
cd PMAccelerator-DataScience

# 2. Setup virtual environment & install requirements
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Execute the full pipeline
python src/data_cleaning.py
python src/02_eda.py
python src/03_forecasting.py
python src/04_advanced_analysis.py
```

---

## 1. Data Preprocessing (`src/data_cleaning.py`)
The raw dataset was processed to ensure consistency and prevent bias in the models:
*   **Redundancy Removal:** Dropped duplicate imperial units (Fahrenheit) to focus on metric (Celsius) variance.
*   **Handling Nulls & Placeholders:** Replaced non-numeric strings (e.g., "No moonset") and placeholder values (e.g., -9999) with `NaN` for appropriate statistical treatment.
*   **Outlier Management:** Filtered physically improbable records (e.g., extreme wind and pressure anomalies) and applied capping to precipitation spikes to maintain regression scale.
*   **Standardization:** Normalized categorical strings for consistent grouping across geographic locations.

## 2. Exploratory Data Analysis (`src/02_eda.py`)
Statistical exploration of the features included:
*   **Distribution Analysis:** Comparison between right-skewed precipitation patterns and normal temperature distributions.
*   **Anomaly Detection:** Implementation of the Interquartile Range (IQR) method to isolate and map abnormal weather occurrences.
*   **Correlation Mapping:** Analysis of linear relationships between humidity, wind speed, and various air quality indexes.

## 3. Modeling & Forecasting (`src/03_forecasting.py`)
The objective was to forecast **Temperature (°C)** trends using an ensemble of models with different architectural approaches. Performance was evaluated using RMSE, MAE, and MAPE.

**Models Implemented:**
*   **Time-Series:** `SARIMA` and `Prophet` for seasonal pattern detection.
*   **Linear Regression:** To establish a baseline for macro trends.
*   **Gradient Boosting:** To capture non-linear interactions between weather variables.
*   **Weighted Ensemble:** A combined prediction model designed to minimize individual algorithm bias and improve generalization.

## 4. Advanced Analytical Modules (`src/04_advanced_analysis.py`)
Extended analysis beyond basic forecasting:
*   **Hemispheric Comparison:** Analysis of temperature and precipitation inversion between Northern and Southern hemispheres.
*   **Environmental Correlations:** Evaluation of weather impacts on air quality (e.g., wind-driven dispersion of PM2.5 and temperature-dependent Ozone levels).
*   **Feature Importance:** Ranking of predictive variables using Pearson Correlation, Permutation Importance, and Tree-Impurity methods.
*   **Geospatial Visualization:** Mapping of **439 unique weather stations** using latitude/longitude data to visualize global UV index, temperature, and pollution distributions.
*   **Climate Baselines:** Identification of extreme values (hottest/coldest) and average precipitation patterns across continents.
---
