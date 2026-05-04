"""
Model Comparison Report — combine regression and classification results
into a unified comparison table + grouped bar chart.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("outputs/reports", exist_ok=True)

# Load results
reg = pd.read_csv("outputs/reports/regression_results.csv")
clf = pd.read_csv("outputs/reports/classification_results.csv")

print("=== REGRESSION RESULTS ===")
print(reg.to_string(index=False))

print("\n=== CLASSIFICATION RESULTS ===")
print(clf.to_string(index=False))

# --- Grouped Bar Charts ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Regression comparison
tasks_r = reg["task"].unique()
x = np.arange(len(tasks_r))
width = 0.35
rf_r2 = reg[reg["model"] == "RF"]["test_r2"].values
xgb_r2 = reg[reg["model"] == "XGBoost"]["test_r2"].values

axes[0].bar(x - width/2, rf_r2, width, label="RF", color="#4C72B0", edgecolor="black")
axes[0].bar(x + width/2, xgb_r2, width, label="XGBoost", color="#DD8452", edgecolor="black")
axes[0].set_xticks(x)
axes[0].set_xticklabels([t.replace(" regression", "") for t in tasks_r], fontsize=10)
axes[0].set_ylabel("Test R²")
axes[0].set_title("Regression: RF vs XGBoost (R²)")
axes[0].legend()
axes[0].set_ylim(0, 1)
for i, (r, xg) in enumerate(zip(rf_r2, xgb_r2)):
    axes[0].text(i - width/2, r + 0.01, f"{r:.3f}", ha="center", fontsize=9)
    axes[0].text(i + width/2, xg + 0.01, f"{xg:.3f}", ha="center", fontsize=9)

# Classification comparison
tasks_c = clf["task"].unique()
x = np.arange(len(tasks_c))
rf_acc = clf[clf["model"] == "RF"]["test_accuracy"].values
xgb_acc = clf[clf["model"] == "XGBoost"]["test_accuracy"].values

axes[1].bar(x - width/2, rf_acc, width, label="RF", color="#4C72B0", edgecolor="black")
axes[1].bar(x + width/2, xgb_acc, width, label="XGBoost", color="#DD8452", edgecolor="black")
axes[1].set_xticks(x)
axes[1].set_xticklabels(tasks_c, fontsize=10)
axes[1].set_ylabel("Test Accuracy")
axes[1].set_title("Classification: RF vs XGBoost (Accuracy)")
axes[1].legend()
axes[1].set_ylim(0, 1)
for i, (r, xg) in enumerate(zip(rf_acc, xgb_acc)):
    axes[1].text(i - width/2, r + 0.01, f"{r:.3f}", ha="center", fontsize=9)
    axes[1].text(i + width/2, xg + 0.01, f"{xg:.3f}", ha="center", fontsize=9)

plt.suptitle("Model Comparison: Random Forest vs XGBoost", fontsize=14)
plt.tight_layout()
plt.savefig("outputs/reports/model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: outputs/reports/model_comparison.png")

# Combine into one CSV
combined = pd.concat([
    reg.rename(columns={"test_r2": "primary_metric", "test_rmse": "secondary_metric"}),
    clf.rename(columns={"test_accuracy": "primary_metric", "test_f1_weighted": "secondary_metric"}),
], ignore_index=True)
combined.to_csv("outputs/reports/model_comparison.csv", index=False)
print("Saved: outputs/reports/model_comparison.csv")
