"""
Exploratory Data Analysis — generate distribution plots, correlation heatmap,
and class balance analysis. Saves all outputs to outputs/reports/eda/.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("outputs/reports/eda", exist_ok=True)

df = pd.read_csv("outputs/reports/cleaned.csv")
print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

# --- Numeric Feature Distributions ---
numeric_cols = [
    "avg_hardness", "avg_density",
    "mend_atomic_mass", "mend_electronegativity", "mend_covalent_radius",
    "mend_ionization_energy", "mend_melting_point", "mend_dipole_polarizability",
    "rdkit_exact_mw", "rdkit_valence_electrons", "rdkit_heavy_atoms",
    "total_unique_elements", "total_atom_count",
    "formula_weighted_mass", "formula_weighted_en",
]

fig, axes = plt.subplots(5, 3, figsize=(18, 20))
axes = axes.flatten()
for i, col in enumerate(numeric_cols):
    if i < len(axes):
        ax = axes[i]
        data = df[col].dropna()
        ax.hist(data, bins=40, edgecolor="black", alpha=0.7, color="#4C72B0")
        ax.set_title(col, fontsize=11)
        ax.set_xlabel("")
        mean_val = data.mean()
        ax.axvline(mean_val, color="red", linestyle="--", alpha=0.7, label=f"mean={mean_val:.2f}")
        ax.legend(fontsize=8)
# Hide unused subplots
for j in range(len(numeric_cols), len(axes)):
    axes[j].set_visible(False)
plt.suptitle("Numeric Feature Distributions", fontsize=16, y=1.01)
plt.tight_layout()
plt.savefig("outputs/reports/eda/numeric_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: numeric_distributions.png")

# --- Categorical Feature Bar Charts ---
cat_cols = ["crystal_system", "transparency", "industry_application"]
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for i, col in enumerate(cat_cols):
    vc = df[col].value_counts().head(10)
    axes[i].barh(vc.index[::-1], vc.values[::-1], color="#4C72B0", edgecolor="black")
    axes[i].set_title(f"{col} (top 10)", fontsize=12)
    for j, v in enumerate(vc.values[::-1]):
        axes[i].text(v + 5, j, str(v), va="center", fontsize=9)
plt.suptitle("Categorical Feature Distributions", fontsize=14)
plt.tight_layout()
plt.savefig("outputs/reports/eda/categorical_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: categorical_distributions.png")

# --- Correlation Heatmap ---
corr_cols = [
    "avg_hardness", "avg_density",
    "mend_atomic_mass", "mend_electronegativity", "mend_covalent_radius",
    "mend_ionization_energy", "mend_melting_point", "mend_dipole_polarizability",
    "rdkit_exact_mw", "rdkit_valence_electrons", "rdkit_heavy_atoms",
    "formula_weighted_mass", "formula_weighted_en",
]
corr = df[corr_cols].corr()

plt.figure(figsize=(14, 11))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, square=True, linewidths=0.5,
            cbar_kws={"shrink": 0.8})
plt.title("Feature Correlation Matrix", fontsize=14)
plt.tight_layout()
plt.savefig("outputs/reports/eda/correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: correlation_heatmap.png")

# --- Class Balance Analysis ---
print("\n=== Class Balance ===")

# Transparency (simplified)
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
print("\nTransparency (simplified):")
vc_t = df["transparency_simple"].value_counts()
for label, count in vc_t.items():
    print(f"  {label}: {count} ({100*count/len(df):.1f}%)")

print("\nCrystal System:")
vc_c = df["crystal_system"].value_counts()
for label, count in vc_c.items():
    print(f"  {label}: {count} ({100*count/len(df):.1f}%)")

print("\nIndustry Application:")
vc_i = df["industry_application"].value_counts()
for label, count in vc_i.items():
    print(f"  {label}: {count} ({100*count/len(df):.1f}%)")

# Class balance pie charts
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, (title, vc) in zip(axes, [
    ("Transparency (Simplified)", df["transparency_simple"].value_counts()),
    ("Crystal System", df["crystal_system"].value_counts()),
    ("Industry Application", df["industry_application"].value_counts()),
]):
    colors = plt.cm.Set3(np.linspace(0, 1, len(vc)))
    ax.pie(vc.values, labels=vc.index, autopct="%1.1f%%", colors=colors, startangle=90)
    ax.set_title(title, fontsize=12)

plt.suptitle("Class Balance Analysis", fontsize=14)
plt.tight_layout()
plt.savefig("outputs/reports/eda/class_balance.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: class_balance.png")

# --- Summary Statistics ---
print("\n=== Summary Statistics ===")
print(df[numeric_cols].describe().round(2).to_string())
