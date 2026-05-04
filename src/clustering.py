"""
Unsupervised Learning — K-Means, Hierarchical clustering, PCA / t-SNE visualization.
Discovers natural mineral groupings and saves cluster assignments + plots.
"""
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage

os.makedirs("outputs/reports/clustering", exist_ok=True)

df = pd.read_csv("outputs/reports/cleaned.csv")

num_features = [
    "mend_atomic_mass", "mend_electronegativity", "mend_covalent_radius",
    "mend_ionization_energy", "mend_melting_point", "mend_dipole_polarizability",
    "rdkit_exact_mw", "rdkit_valence_electrons", "rdkit_heavy_atoms",
    "avg_hardness", "avg_density",
    "total_unique_elements", "total_atom_count",
    "formula_weighted_mass", "formula_weighted_en",
]
cat_features = ["crystal_system", "transparency", "industry_application"]

# Preprocessing: impute + scale numerics, OHE categoricals
preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), num_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_features),
    ]
)

X_raw = df[num_features + cat_features]
X_scaled = preprocess.fit_transform(X_raw)
print(f"Scaled feature matrix: {X_scaled.shape}")

# ============================================================
# K-MEANS CLUSTERING
# ============================================================
print("\n=== K-Means Clustering ===")

K_range = range(2, 13)
inertias = []
silhouettes = []

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_scaled, labels, sample_size=min(2000, len(X_scaled)))
    silhouettes.append(sil)
    print(f"  k={k}: inertia={km.inertia_:.0f}, silhouette={sil:.4f}")

# Elbow + Silhouette plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(list(K_range), inertias, 'bo-', linewidth=2)
ax1.set_xlabel("Number of Clusters (k)")
ax1.set_ylabel("Inertia")
ax1.set_title("Elbow Method")
ax1.grid(True, alpha=0.3)

ax2.plot(list(K_range), silhouettes, 'ro-', linewidth=2)
ax2.set_xlabel("Number of Clusters (k)")
ax2.set_ylabel("Silhouette Score")
ax2.set_title("Silhouette Analysis")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/reports/clustering/elbow_silhouette.png", dpi=150)
plt.close()
print("Saved: elbow_silhouette.png")

# Choose optimal k (best silhouette)
best_k = list(K_range)[np.argmax(silhouettes)]
print(f"\nOptimal k by silhouette: {best_k}")

# Final K-Means with optimal k
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["cluster_kmeans"] = km_final.fit_predict(X_scaled)

# Cluster composition analysis
print(f"\n=== Cluster Composition (k={best_k}) ===")
for c in range(best_k):
    subset = df[df["cluster_kmeans"] == c]
    print(f"\nCluster {c} ({len(subset)} minerals):")
    print(f"  Top crystal systems: {subset['crystal_system'].value_counts().head(3).to_dict()}")
    print(f"  Top industries: {subset['industry_application'].value_counts().head(3).to_dict()}")
    print(f"  Avg hardness: {subset['avg_hardness'].mean():.2f}")
    print(f"  Avg density: {subset['avg_density'].mean():.2f}")

# ============================================================
# HIERARCHICAL CLUSTERING
# ============================================================
print("\n=== Hierarchical Clustering ===")

# Use a sample for dendrogram (too slow with full dataset)
np.random.seed(42)
sample_idx = np.random.choice(len(X_scaled), size=min(200, len(X_scaled)), replace=False)
X_sample = X_scaled[sample_idx]

Z = linkage(X_sample, method="ward")

plt.figure(figsize=(16, 8))
dendrogram(Z, truncate_mode="lastp", p=30, leaf_rotation=90, leaf_font_size=8)
plt.title("Hierarchical Clustering Dendrogram (Ward, 200 sample)")
plt.xlabel("Sample Index or (cluster size)")
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig("outputs/reports/clustering/dendrogram.png", dpi=150)
plt.close()
print("Saved: dendrogram.png")

# Agglomerative clustering on full dataset
agg = AgglomerativeClustering(n_clusters=best_k)
df["cluster_hierarchical"] = agg.fit_predict(X_scaled)

# ============================================================
# PCA & t-SNE VISUALIZATION
# ============================================================
print("\n=== Dimensionality Reduction ===")

# PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
df["pca_1"] = X_pca[:, 0]
df["pca_2"] = X_pca[:, 1]
print(f"PCA explained variance: {pca.explained_variance_ratio_}")
print(f"PCA total variance explained: {pca.explained_variance_ratio_.sum():.4f}")

# t-SNE (on a manageable subset or full)
print("Running t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)
df["tsne_1"] = X_tsne[:, 0]
df["tsne_2"] = X_tsne[:, 1]

# --- Plot PCA colored by different attributes ---
color_by = {
    "cluster_kmeans": ("K-Means Cluster", "tab10"),
    "crystal_system": ("Crystal System", "tab10"),
    "industry_application": ("Industry", "Set2"),
}

for col, (title, cmap) in color_by.items():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    categories = df[col].astype(str).unique()
    color_map = {cat: plt.cm.get_cmap(cmap)(i / max(len(categories)-1, 1))
                 for i, cat in enumerate(categories)}

    for cat in categories:
        mask = df[col].astype(str) == cat
        ax1.scatter(df.loc[mask, "pca_1"], df.loc[mask, "pca_2"],
                    c=[color_map[cat]], label=cat, alpha=0.5, s=10)
        ax2.scatter(df.loc[mask, "tsne_1"], df.loc[mask, "tsne_2"],
                    c=[color_map[cat]], label=cat, alpha=0.5, s=10)

    ax1.set_title(f"PCA — Colored by {title}")
    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax1.legend(fontsize=8, markerscale=3, loc="best")

    ax2.set_title(f"t-SNE — Colored by {title}")
    ax2.set_xlabel("t-SNE 1")
    ax2.set_ylabel("t-SNE 2")
    ax2.legend(fontsize=8, markerscale=3, loc="best")

    plt.tight_layout()
    fname = f"outputs/reports/clustering/projection_{col}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved: {fname}")

# Save updated dataset with cluster labels and projections
df.to_csv("outputs/reports/cleaned.csv", index=False)
print("\nSaved updated cleaned.csv with cluster labels + PCA/t-SNE coordinates")

# Save clustering summary
cluster_summary = df.groupby("cluster_kmeans").agg(
    count=("name", "count"),
    avg_hardness=("avg_hardness", "mean"),
    avg_density=("avg_density", "mean"),
    top_crystal=("crystal_system", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "N/A"),
    top_industry=("industry_application", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "N/A"),
).round(2)
cluster_summary.to_csv("outputs/reports/clustering/cluster_summary.csv")
print("Saved: cluster_summary.csv")
print(cluster_summary.to_string())
