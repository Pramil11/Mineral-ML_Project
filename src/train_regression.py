"""
Train ALL regression models (RF + XGBoost) for avg_hardness and avg_density.
Includes 5-fold cross-validation and saves results to CSV.
"""
import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

os.makedirs("outputs/models", exist_ok=True)

df = pd.read_csv("outputs/reports/cleaned.csv")

num_features = [
    "mend_atomic_mass", "mend_electronegativity", "mend_covalent_radius",
    "mend_ionization_energy", "mend_melting_point", "mend_dipole_polarizability",
    "rdkit_exact_mw", "rdkit_valence_electrons", "rdkit_heavy_atoms",
    "total_unique_elements", "total_atom_count",
    "formula_weighted_mass", "formula_weighted_en",
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

X = df[num_features + cat_features]

results = []

for target_name, y in [("avg_hardness", df["avg_hardness"]), ("avg_density", df["avg_density"])]:
    print(f"\n{'='*60}")
    print(f"Target: {target_name}")
    print(f"{'='*60}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "RF": Pipeline([("prep", preprocess), ("model", RandomForestRegressor(
            n_estimators=500, random_state=42, n_jobs=-1
        ))]),
        "XGBoost": Pipeline([("prep", preprocess), ("model", XGBRegressor(
            n_estimators=500, random_state=42, learning_rate=0.1,
            max_depth=6, n_jobs=-1, verbosity=0
        ))]),
    }

    for model_name, pipe in models.items():
        print(f"\n--- {model_name} ---")

        # 5-fold cross-validation
        cv_r2 = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
        cv_rmse = -cross_val_score(pipe, X_train, y_train, cv=5,
                                    scoring="neg_root_mean_squared_error", n_jobs=-1)

        print(f"  CV R²:   {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
        print(f"  CV RMSE: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")

        # Final fit on full training set
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        test_r2 = r2_score(y_test, pred)
        test_rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        print(f"  Test R²:   {test_r2:.4f}")
        print(f"  Test RMSE: {test_rmse:.4f}")

        # Save model
        model_file = f"outputs/models/{model_name.lower()}_reg_{target_name}.pkl"
        # Rename RF models to match expected naming
        if model_name == "RF":
            model_file = f"outputs/models/rf_reg_{target_name}.pkl"
        elif model_name == "XGBoost":
            model_file = f"outputs/models/xgb_reg_{target_name}.pkl"
        joblib.dump(pipe, model_file)
        print(f"  Saved: {model_file}")

        results.append({
            "task": f"{target_name} regression",
            "model": model_name,
            "cv_r2_mean": round(cv_r2.mean(), 4),
            "cv_r2_std": round(cv_r2.std(), 4),
            "cv_rmse_mean": round(cv_rmse.mean(), 4),
            "cv_rmse_std": round(cv_rmse.std(), 4),
            "test_r2": round(test_r2, 4),
            "test_rmse": round(test_rmse, 4),
        })

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("outputs/reports/regression_results.csv", index=False)
print("\nSaved: outputs/reports/regression_results.csv")
print(results_df.to_string(index=False))