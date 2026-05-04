"""
Train ALL classification models (RF + XGBoost) for:
- Transparency (3-class: Transparent/Translucent/Opaque)
- Crystal System
- Industry Application
Includes 5-fold stratified cross-validation.
"""
import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from xgboost import XGBClassifier

os.makedirs("outputs/models", exist_ok=True)

df = pd.read_csv("outputs/reports/cleaned.csv")

num_features = [
    "mend_atomic_mass", "mend_electronegativity", "mend_covalent_radius",
    "mend_ionization_energy", "mend_melting_point", "mend_dipole_polarizability",
    "rdkit_exact_mw", "rdkit_valence_electrons", "rdkit_heavy_atoms",
    "total_unique_elements", "total_atom_count",
    "formula_weighted_mass", "formula_weighted_en",
]

def simplify_transparency(x):
    x = str(x)
    if x.strip() == "" or x.lower() == "nan" or x == "Unknown":
        return "Unknown"
    if "Transparent" in x:
        return "Transparent"
    if "Translucent" in x:
        return "Translucent"
    if "Opaque" in x:
        return "Opaque"
    return "Unknown"


def make_preprocessor(cat_cols):
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_features),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols),
        ]
    )


results = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ============================================================
# TASK 1: Transparency Classification (3-class)
# ============================================================
print("=" * 60)
print("TRANSPARENCY CLASSIFICATION (3-class)")
print("=" * 60)

df_t = df.copy()
df_t["transparency_simple"] = df_t["transparency"].apply(simplify_transparency)
df_t = df_t[df_t["transparency_simple"] != "Unknown"].copy()

cat_features_t = ["crystal_system", "cleavage", "industry_application"]
X_t = df_t[num_features + cat_features_t]
y_t = df_t["transparency_simple"]

X_train, X_test, y_train, y_test = train_test_split(X_t, y_t, test_size=0.2, random_state=42, stratify=y_t)

models_t = {
    "RF": Pipeline([("prep", make_preprocessor(cat_features_t)), ("model", RandomForestClassifier(
        n_estimators=600, random_state=42, n_jobs=-1, class_weight="balanced"
    ))]),
    "XGBoost": Pipeline([("prep", make_preprocessor(cat_features_t)), ("model", XGBClassifier(
        n_estimators=600, random_state=42, learning_rate=0.1, max_depth=6,
        n_jobs=-1, verbosity=0, eval_metric="mlogloss",
        use_label_encoder=False
    ))]),
}

# XGBoost needs numeric labels
le_t = LabelEncoder()
y_train_enc = le_t.fit_transform(y_train)
y_test_enc = le_t.transform(y_test)

for model_name, pipe in models_t.items():
    print(f"\n--- {model_name} ---")

    if model_name == "XGBoost":
        yt_train, yt_test = y_train_enc, y_test_enc
    else:
        yt_train, yt_test = y_train, y_test

    cv_acc = cross_val_score(pipe, X_train, yt_train, cv=skf, scoring="accuracy", n_jobs=-1)
    print(f"  CV Accuracy: {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")

    pipe.fit(X_train, yt_train)
    pred = pipe.predict(X_test)

    if model_name == "XGBoost":
        pred_labels = le_t.inverse_transform(pred)
    else:
        pred_labels = pred

    acc = accuracy_score(y_test, pred_labels)
    f1 = f1_score(y_test, pred_labels, average="weighted")
    print(f"  Test Accuracy: {acc:.4f}")
    print(f"  Test F1 (weighted): {f1:.4f}")
    print(classification_report(y_test, pred_labels))

    model_file = f"outputs/models/{model_name.lower()}_clf_transparency_simple.pkl"
    joblib.dump(pipe, model_file)
    # For XGBoost, also save the label encoder
    if model_name == "XGBoost":
        joblib.dump(le_t, "outputs/models/xgb_clf_transparency_label_encoder.pkl")
    print(f"  Saved: {model_file}")

    results.append({
        "task": "transparency (3-class)",
        "model": model_name,
        "cv_accuracy_mean": round(cv_acc.mean(), 4),
        "cv_accuracy_std": round(cv_acc.std(), 4),
        "test_accuracy": round(acc, 4),
        "test_f1_weighted": round(f1, 4),
    })


# ============================================================
# TASK 2: Crystal System Classification
# ============================================================
print("\n" + "=" * 60)
print("CRYSTAL SYSTEM CLASSIFICATION")
print("=" * 60)

df_c = df[df["crystal_system"] != "Unknown"].copy()
df_c = df_c[df_c["crystal_system"] != "Amorphous"].copy()  # too few samples

cat_features_c = ["cleavage", "transparency", "industry_application"]
X_c = df_c[num_features + cat_features_c]
y_c = df_c["crystal_system"]

X_train, X_test, y_train, y_test = train_test_split(X_c, y_c, test_size=0.2, random_state=42, stratify=y_c)

le_c = LabelEncoder()
y_train_enc = le_c.fit_transform(y_train)
y_test_enc = le_c.transform(y_test)

models_c = {
    "RF": Pipeline([("prep", make_preprocessor(cat_features_c)), ("model", RandomForestClassifier(
        n_estimators=500, random_state=42, n_jobs=-1, class_weight="balanced"
    ))]),
    "XGBoost": Pipeline([("prep", make_preprocessor(cat_features_c)), ("model", XGBClassifier(
        n_estimators=500, random_state=42, learning_rate=0.1, max_depth=6,
        n_jobs=-1, verbosity=0, eval_metric="mlogloss",
        use_label_encoder=False
    ))]),
}

for model_name, pipe in models_c.items():
    print(f"\n--- {model_name} ---")

    if model_name == "XGBoost":
        yt_train, yt_test = y_train_enc, y_test_enc
    else:
        yt_train, yt_test = y_train, y_test

    cv_acc = cross_val_score(pipe, X_train, yt_train, cv=skf, scoring="accuracy", n_jobs=-1)
    print(f"  CV Accuracy: {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")

    pipe.fit(X_train, yt_train)
    pred = pipe.predict(X_test)

    if model_name == "XGBoost":
        pred_labels = le_c.inverse_transform(pred)
    else:
        pred_labels = pred

    acc = accuracy_score(y_test, pred_labels)
    f1 = f1_score(y_test, pred_labels, average="weighted")
    print(f"  Test Accuracy: {acc:.4f}")
    print(f"  Test F1 (weighted): {f1:.4f}")
    print(classification_report(y_test, pred_labels))

    model_file = f"outputs/models/{model_name.lower()}_clf_crystal_system.pkl"
    joblib.dump(pipe, model_file)
    if model_name == "XGBoost":
        joblib.dump(le_c, "outputs/models/xgb_clf_crystal_label_encoder.pkl")
    print(f"  Saved: {model_file}")

    results.append({
        "task": "crystal_system",
        "model": model_name,
        "cv_accuracy_mean": round(cv_acc.mean(), 4),
        "cv_accuracy_std": round(cv_acc.std(), 4),
        "test_accuracy": round(acc, 4),
        "test_f1_weighted": round(f1, 4),
    })


# ============================================================
# TASK 3: Industry Classification
# ============================================================
print("\n" + "=" * 60)
print("INDUSTRY CLASSIFICATION")
print("=" * 60)

df_i = df[df["industry_application"] != "Unknown"].copy()

cat_features_i = ["crystal_system", "cleavage", "transparency"]
X_i = df_i[num_features + cat_features_i]
y_i = df_i["industry_application"]

X_train, X_test, y_train, y_test = train_test_split(X_i, y_i, test_size=0.2, random_state=42, stratify=y_i)

le_i = LabelEncoder()
y_train_enc = le_i.fit_transform(y_train)
y_test_enc = le_i.transform(y_test)

models_i = {
    "RF": Pipeline([("prep", make_preprocessor(cat_features_i)), ("model", RandomForestClassifier(
        n_estimators=500, random_state=42, n_jobs=-1, class_weight="balanced"
    ))]),
    "XGBoost": Pipeline([("prep", make_preprocessor(cat_features_i)), ("model", XGBClassifier(
        n_estimators=500, random_state=42, learning_rate=0.1, max_depth=6,
        n_jobs=-1, verbosity=0, eval_metric="mlogloss",
        use_label_encoder=False
    ))]),
}

for model_name, pipe in models_i.items():
    print(f"\n--- {model_name} ---")

    if model_name == "XGBoost":
        yt_train, yt_test = y_train_enc, y_test_enc
    else:
        yt_train, yt_test = y_train, y_test

    cv_acc = cross_val_score(pipe, X_train, yt_train, cv=skf, scoring="accuracy", n_jobs=-1)
    print(f"  CV Accuracy: {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")

    pipe.fit(X_train, yt_train)
    pred = pipe.predict(X_test)

    if model_name == "XGBoost":
        pred_labels = le_i.inverse_transform(pred)
    else:
        pred_labels = pred

    acc = accuracy_score(y_test, pred_labels)
    f1 = f1_score(y_test, pred_labels, average="weighted")
    print(f"  Test Accuracy: {acc:.4f}")
    print(f"  Test F1 (weighted): {f1:.4f}")
    print(classification_report(y_test, pred_labels))

    model_file = f"outputs/models/{model_name.lower()}_clf_industry.pkl"
    joblib.dump(pipe, model_file)
    if model_name == "XGBoost":
        joblib.dump(le_i, "outputs/models/xgb_clf_industry_label_encoder.pkl")
    print(f"  Saved: {model_file}")

    results.append({
        "task": "industry",
        "model": model_name,
        "cv_accuracy_mean": round(cv_acc.mean(), 4),
        "cv_accuracy_std": round(cv_acc.std(), 4),
        "test_accuracy": round(acc, 4),
        "test_f1_weighted": round(f1, 4),
    })


# Save all classification results
results_df = pd.DataFrame(results)
results_df.to_csv("outputs/reports/classification_results.csv", index=False)
print("\n" + "=" * 60)
print("ALL CLASSIFICATION RESULTS")
print("=" * 60)
print(results_df.to_string(index=False))