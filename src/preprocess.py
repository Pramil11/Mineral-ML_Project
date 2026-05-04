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

df["transparency"] = df["transparency"].fillna("Unknown").astype(str).str.strip()
df["crystal_system"] = df["crystal_system"].fillna("Unknown").astype(str).str.strip()
df["industry_application"] = df["industry_application"].fillna("Unknown").astype(str).str.strip()

for c in ["avg_hardness", "avg_density"]:
    df.loc[~np.isfinite(df[c]), c] = np.nan

df = df.dropna(subset=["avg_hardness", "avg_density"]).copy()
final_rows = len(df)

df.to_csv("outputs/reports/cleaned.csv", index=False)

try:
    df.to_parquet("outputs/reports/cleaned.parquet", index=False)
except Exception:
    pass

print("Saved cleaned dataset:")
print("outputs/reports/cleaned.csv")
print("Rows:", final_rows)
print("Dropped rows (missing avg_hardness/avg_density):", original_rows - final_rows)