import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import numpy as np

os.makedirs("outputs/models", exist_ok=True)

df = pd.read_csv("outputs/reports/cleaned.csv")

num_features = [
    "mend_atomic_mass",
    "mend_electronegativity",
    "mend_covalent_radius",
    "mend_ionization_energy",
    "mend_melting_point",
    "mend_dipole_polarizability",
    "rdkit_exact_mw",
    "rdkit_valence_electrons",
    "rdkit_heavy_atoms",
]
cat_features = ["crystal_system", "cleavage", "transparency", "industry_application"]

preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_features),
    ]
)

# --- Train Hardness Regression ---
X = df[num_features + cat_features]
y_h = df["avg_hardness"]

X_train, X_test, y_train, y_test = train_test_split(X, y_h, test_size=0.2, random_state=42)

reg_h = Pipeline([("prep", preprocess), ("model", RandomForestRegressor(
    n_estimators=500, random_state=42, n_jobs=-1
))])
reg_h.fit(X_train, y_train)

pred_h = reg_h.predict(X_test)
print(f"Hardness Regression: RMSE={np.sqrt(mean_squared_error(y_test, pred_h)):.4f}, R2={r2_score(y_test, pred_h):.4f}")

joblib.dump(reg_h, "outputs/models/rf_reg_avg_hardness.pkl")
print("Saved: outputs/models/rf_reg_avg_hardness.pkl")

# --- Train Density Regression ---
y_d = df["avg_density"]

X_train, X_test, y_train, y_test = train_test_split(X, y_d, test_size=0.2, random_state=42)

reg_d = Pipeline([("prep", preprocess), ("model", RandomForestRegressor(
    n_estimators=500, random_state=42, n_jobs=-1
))])
reg_d.fit(X_train, y_train)

pred_d = reg_d.predict(X_test)
print(f"Density Regression: RMSE={np.sqrt(mean_squared_error(y_test, pred_d)):.4f}, R2={r2_score(y_test, pred_d):.4f}")

joblib.dump(reg_d, "outputs/models/rf_reg_avg_density.pkl")
print("Saved: outputs/models/rf_reg_avg_density.pkl")