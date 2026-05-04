# Mineral-ML Project

A complete Data Science and Machine Learning project to predict mineral properties using advanced chemical and structural features.

## Overview
This project predicts **Hardness**, **Density**, and **Transparency**, as well as **Crystal System** and **Industry Application**, using both Random Forest and XGBoost models.
The dataset incorporates features from Mindat.org, enhanced with element-level properties extracted using RDKit and Mendeleev libraries.

## Features
* **Feature Engineering:** Parses chemical formulas (`SiO2` → `{Si: 1, O: 2}`) and calculates weighted properties (mass, electronegativity).
* **EDA & Visualization:** Generates correlations, distributions, and class balances.
* **Unsupervised Learning:** Implements K-Means and Hierarchical clustering with PCA and t-SNE 2D visualizations.
* **Predictive Modeling:** XGBoost and Random Forest pipelines with `ColumnTransformer` for categorical and numerical handling.
* **Dashboard:** An interactive Streamlit dashboard allowing you to filter data, visualize clusters and feature importances, and run live model predictions.

## Setup Instructions

1. Ensure you have Python 3.12+ installed.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Execution Order

Run the following scripts in order to completely rebuild the data, models, and reports:

1. **Preprocess & Feature Engineering:**
   ```bash
   python src/preprocess.py
   python src/parse_formulas.py
   ```
2. **Exploratory Data Analysis:**
   ```bash
   python src/eda.py
   ```
3. **Model Training (RF & XGBoost):**
   ```bash
   python src/train_regression.py
   python src/train_classification.py
   ```
4. **Feature Importance & Model Comparison:**
   ```bash
   python src/feature_importance.py
   python src/model_comparison.py
   ```
5. **Clustering & Dimensionality Reduction:**
   ```bash
   python src/clustering.py
   ```

## Dashboard
Run the Streamlit application to explore the models interactively:
```bash
streamlit run app/streamlit_app.py
```