import os
import pandas as pd
import numpy as np

os.makedirs("outputs/reports", exist_ok=True)

df = pd.read_csv("data/enhanced_assignment3_dataset.csv", low_memory=False)

original_rows = len(df)

num_cols = ["hardness_min", "hardness_max", "density_min", "density_max"]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["avg_hardness"] = (df["hardness_min"] + df["hardness_max"]) / 2
df["avg_density"] = (df["density_min"] + df["density_max"]) / 2

# --- Task 1.1: Outlier Handling ---
outlier_hardness = (df["avg_hardness"] > 10) | (df["avg_hardness"] < 0)
outlier_density = (df["avg_density"] < 0) | (df["avg_density"] > 25)
n_outlier_hardness = outlier_hardness.sum()
n_outlier_density = outlier_density.sum()

df.loc[outlier_hardness, "avg_hardness"] = np.nan
df.loc[outlier_density, "avg_density"] = np.nan

print(f"Outliers removed — hardness out of [0,10]: {n_outlier_hardness}, density out of [0,25]: {n_outlier_density}")

# --- Task 1.2: Crystal System Normalization ---
CRYSTAL_SYSTEM_MAP = {
    "Isometric": "Cubic",
    "isometric": "Cubic",
    "cubic": "Cubic",
    "Cubic": "Cubic",
}

df["transparency"] = df["transparency"].fillna("Unknown").astype(str).str.strip()
df["crystal_system"] = df["crystal_system"].fillna("Unknown").astype(str).str.strip()
df["crystal_system"] = df["crystal_system"].replace(CRYSTAL_SYSTEM_MAP)
df["industry_application"] = df["industry_application"].fillna("Unknown").astype(str).str.strip()
df["cleavage"] = df["cleavage"].fillna("Unknown").astype(str).str.strip()

print(f"Crystal system values after normalization: {df['crystal_system'].nunique()} unique")
print(df["crystal_system"].value_counts().to_string())

for c in ["avg_hardness", "avg_density"]:
    df.loc[~np.isfinite(df[c]), c] = np.nan

df = df.dropna(subset=["avg_hardness", "avg_density"]).copy()
final_rows = len(df)

df.to_csv("outputs/reports/cleaned.csv", index=False)

try:
    df.to_parquet("outputs/reports/cleaned.parquet", index=False)
except Exception:
    pass

print("\nSaved cleaned dataset:")
print("outputs/reports/cleaned.csv")
print("Rows:", final_rows)
print("Dropped rows (missing avg_hardness/avg_density):", original_rows - final_rows)