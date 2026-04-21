import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

os.makedirs("outputs/models", exist_ok=True)

df = pd.read_csv("outputs/reports/cleaned.csv")

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

df["transparency_simple"] = df["transparency"].apply(simplify_transparency)
df = df[df["transparency_simple"] != "Unknown"].copy()

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
cat_features = ["crystal_system", "cleavage", "industry_application"]

X = df[num_features + cat_features]
y = df["transparency_simple"]

preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_features),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(
    n_estimators=600,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

pipe = Pipeline([("prep", preprocess), ("model", clf)])
pipe.fit(X_train, y_train)

pred = pipe.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print("\nReport:\n", classification_report(y_test, pred))

joblib.dump(pipe, "outputs/models/rf_clf_transparency_simple.pkl")
print("Saved: outputs/models/rf_clf_transparency_simple.pkl")