"""
Feature Importance Analysis — extract and visualize feature importances
from all trained models (RF + XGBoost).
"""
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("outputs/reports/feature_importance", exist_ok=True)

df = pd.read_csv("outputs/reports/cleaned.csv")

num_features = [
    "mend_atomic_mass", "mend_electronegativity", "mend_covalent_radius",
    "mend_ionization_energy", "mend_melting_point", "mend_dipole_polarizability",
    "rdkit_exact_mw", "rdkit_valence_electrons", "rdkit_heavy_atoms",
    "total_unique_elements", "total_atom_count",
    "formula_weighted_mass", "formula_weighted_en",
]


def get_feature_names(pipe):
    """Extract feature names from a fitted pipeline with ColumnTransformer."""
    ct = pipe.named_steps["prep"]
    feature_names = list(num_features)  # numeric features keep their names
    # Get OHE feature names from the categorical transformer
    try:
        cat_pipe = ct.named_transformers_["cat"]
        ohe = cat_pipe.named_steps["ohe"]
        cat_names = list(ohe.get_feature_names_out())
        feature_names.extend(cat_names)
    except Exception:
        pass
    return feature_names


def extract_importances(pipe, feature_names):
    """Extract feature importances from the model step of a pipeline."""
    model = pipe.named_steps["model"]
    importances = model.feature_importances_
    # Truncate or pad feature names if needed
    n = len(importances)
    if len(feature_names) > n:
        feature_names = feature_names[:n]
    elif len(feature_names) < n:
        feature_names.extend([f"feature_{i}" for i in range(len(feature_names), n)])
    return pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)


def plot_importance_comparison(rf_imp, xgb_imp, title, filename, top_n=15):
    """Plot side-by-side RF vs XGBoost feature importance."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # RF
    top_rf = rf_imp.head(top_n).sort_values("importance")
    ax1.barh(top_rf["feature"], top_rf["importance"], color="#4C72B0", edgecolor="black")
    ax1.set_title(f"Random Forest — {title}", fontsize=12)
    ax1.set_xlabel("Feature Importance")

    # XGBoost
    top_xgb = xgb_imp.head(top_n).sort_values("importance")
    ax2.barh(top_xgb["feature"], top_xgb["importance"], color="#DD8452", edgecolor="black")
    ax2.set_title(f"XGBoost — {title}", fontsize=12)
    ax2.set_xlabel("Feature Importance")

    plt.suptitle(f"Feature Importance: {title}", fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


# === Regression: Hardness ===
rf_h = joblib.load("outputs/models/rf_reg_avg_hardness.pkl")
xgb_h = joblib.load("outputs/models/xgb_reg_avg_hardness.pkl")
fn_h = get_feature_names(rf_h)
rf_h_imp = extract_importances(rf_h, fn_h.copy())
xgb_h_imp = extract_importances(xgb_h, fn_h.copy())
plot_importance_comparison(rf_h_imp, xgb_h_imp, "Hardness Regression",
                           "outputs/reports/feature_importance/hardness_importance.png")

# === Regression: Density ===
rf_d = joblib.load("outputs/models/rf_reg_avg_density.pkl")
xgb_d = joblib.load("outputs/models/xgb_reg_avg_density.pkl")
fn_d = get_feature_names(rf_d)
rf_d_imp = extract_importances(rf_d, fn_d.copy())
xgb_d_imp = extract_importances(xgb_d, fn_d.copy())
plot_importance_comparison(rf_d_imp, xgb_d_imp, "Density Regression",
                           "outputs/reports/feature_importance/density_importance.png")

# === Classification: Transparency ===
rf_t = joblib.load("outputs/models/rf_clf_transparency_simple.pkl")
xgb_t = joblib.load("outputs/models/xgboost_clf_transparency_simple.pkl")
fn_t = get_feature_names(rf_t)
rf_t_imp = extract_importances(rf_t, fn_t.copy())
xgb_t_imp = extract_importances(xgb_t, fn_t.copy())
plot_importance_comparison(rf_t_imp, xgb_t_imp, "Transparency Classification",
                           "outputs/reports/feature_importance/transparency_importance.png")

# === Classification: Industry ===
rf_i = joblib.load("outputs/models/rf_clf_industry.pkl")
xgb_i = joblib.load("outputs/models/xgboost_clf_industry.pkl")
fn_i = get_feature_names(rf_i)
rf_i_imp = extract_importances(rf_i, fn_i.copy())
xgb_i_imp = extract_importances(xgb_i, fn_i.copy())
plot_importance_comparison(rf_i_imp, xgb_i_imp, "Industry Classification",
                           "outputs/reports/feature_importance/industry_importance.png")

# === Save all importance data to CSV ===
all_imp = []
for task, rf_imp, xgb_imp in [
    ("hardness_regression", rf_h_imp, xgb_h_imp),
    ("density_regression", rf_d_imp, xgb_d_imp),
    ("transparency_classification", rf_t_imp, xgb_t_imp),
    ("industry_classification", rf_i_imp, xgb_i_imp),
]:
    for model_name, imp_df in [("RF", rf_imp), ("XGBoost", xgb_imp)]:
        temp = imp_df.copy()
        temp["task"] = task
        temp["model"] = model_name
        all_imp.append(temp)

all_imp_df = pd.concat(all_imp, ignore_index=True)
all_imp_df.to_csv("outputs/reports/feature_importance/all_importances.csv", index=False)
print("Saved: outputs/reports/feature_importance/all_importances.csv")
