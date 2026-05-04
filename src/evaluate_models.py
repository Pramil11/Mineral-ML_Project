import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix, ConfusionMatrixDisplay
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

os.makedirs("outputs/reports", exist_ok=True)

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

X = df[num_features + cat_features]
y_h = df["avg_hardness"]
y_d = df["avg_density"]

reg_h = joblib.load("outputs/models/rf_reg_avg_hardness.pkl")
reg_d = joblib.load("outputs/models/rf_reg_avg_density.pkl")

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def eval_reg(model, y, title, out_png):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pred = model.predict(X_test)

    r2 = r2_score(y_test, pred)
    r = rmse(y_test, pred)
    print(f"{title}: RMSE={r:.4f}, R2={r2:.4f}")

    plt.figure()
    plt.scatter(y_test, pred)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{title} (R2={r2:.2f})")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

eval_reg(reg_h, y_h, "Hardness Regression", "outputs/reports/reg_hardness_actual_vs_pred.png")
eval_reg(reg_d, y_d, "Density Regression", "outputs/reports/reg_density_actual_vs_pred.png")

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
dfc = df[df["transparency_simple"] != "Unknown"].copy()

Xc = dfc[num_features + ["crystal_system", "cleavage", "industry_application"]]
yc = dfc["transparency_simple"]

clf = joblib.load("outputs/models/rf_clf_transparency_simple.pkl")
X_train, X_test, y_train, y_test = train_test_split(Xc, yc, test_size=0.2, random_state=42, stratify=yc)
pred = clf.predict(X_test)

cm = confusion_matrix(y_test, pred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot()
plt.title("Transparency (3-class) Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/reports/clf_transparency_confusion.png", dpi=200)
plt.close()

print("Saved plots to outputs/reports/")