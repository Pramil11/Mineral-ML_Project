import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

st.set_page_config(page_title="Mineral ML Explorer", layout="wide")

# Use st.cache_data to load the main dataset only once
@st.cache_data
def load_data():
    df = pd.read_csv("outputs/reports/cleaned.csv")
    # Simplify transparency if not already done in the CSV
    def simplify_transparency(x):
        x = str(x)
        if x.strip() == "" or x.lower() == "nan" or x == "unknown":
            return "Unknown"
        if "Transparent" in x:
            return "Transparent"
        if "Translucent" in x:
            return "Translucent"
        if "Opaque" in x:
            return "Opaque"
        return "Unknown"
    
    if "transparency_simple" not in df.columns:
        df["transparency_simple"] = df["transparency"].apply(simplify_transparency)
    
    return df

df = load_data()

@st.cache_resource
def load_models():
    models = {}
    try:
        models["rf_reg_h"] = joblib.load("outputs/models/rf_reg_avg_hardness.pkl")
        models["rf_reg_d"] = joblib.load("outputs/models/rf_reg_avg_density.pkl")
        models["rf_clf_t"] = joblib.load("outputs/models/rf_clf_transparency_simple.pkl")
        models["xgb_reg_h"] = joblib.load("outputs/models/xgb_reg_avg_hardness.pkl")
        models["xgb_reg_d"] = joblib.load("outputs/models/xgb_reg_avg_density.pkl")
        models["xgb_clf_t"] = joblib.load("outputs/models/xgboost_clf_transparency_simple.pkl")
        models["xgb_le_t"] = joblib.load("outputs/models/xgb_clf_transparency_label_encoder.pkl")
    except Exception as e:
        st.warning(f"Warning: Could not load all models. Some features may not work. Error: {e}")
    return models

models = load_models()

# Feature lists expected by models
num_features = [
    "mend_atomic_mass", "mend_electronegativity", "mend_covalent_radius",
    "mend_ionization_energy", "mend_melting_point", "mend_dipole_polarizability",
    "rdkit_exact_mw", "rdkit_valence_electrons", "rdkit_heavy_atoms",
    "total_unique_elements", "total_atom_count",
    "formula_weighted_mass", "formula_weighted_en",
]
cat_features_reg = ["crystal_system", "cleavage", "transparency", "industry_application"]
cat_features_clf = ["crystal_system", "cleavage", "industry_application"]

st.title("Mineral ML Dashboard")

tab_explorer, tab_clustering, tab_eda, tab_model_comp, tab_feat_imp = st.tabs([
    "🔍 Data Explorer & Predictions", 
    "🧬 Clustering", 
    "📊 Exploratory Data Analysis", 
    "⚖️ Model Comparison", 
    "⭐ Feature Importance"
])


# ==========================================
# TAB 1: DATA EXPLORER & PREDICTIONS
# ==========================================
with tab_explorer:
    with st.sidebar:
        st.header("Filters")
        
        industry = st.multiselect("Industry", sorted(df["industry_application"].astype(str).unique()))
        crystal = st.multiselect("Crystal system", sorted(df["crystal_system"].astype(str).unique()))
        transparency = st.multiselect("Transparency", sorted(df["transparency_simple"].astype(str).unique()))
        
        # Element filter
        all_elements = set()
        if 'element_list' in df.columns:
            for el_list in df['element_list'].dropna():
                all_elements.update(el_list.split(','))
            all_elements = sorted([e for e in all_elements if e])
            elements = st.multiselect("Contains Elements", all_elements)
        else:
            elements = []

        h_min, h_max = float(df["avg_hardness"].min()), float(df["avg_hardness"].max())
        d_min, d_max = float(df["avg_density"].min()), float(df["avg_density"].max())

        hardness_range = st.slider("Avg hardness", h_min, h_max, (h_min, h_max))
        density_range = st.slider("Avg density", d_min, d_max, (d_min, d_max))

    f = df.copy()
    if industry:
        f = f[f["industry_application"].astype(str).isin(industry)]
    if crystal:
        f = f[f["crystal_system"].astype(str).isin(crystal)]
    if transparency:
        f = f[f["transparency_simple"].astype(str).isin(transparency)]
    if elements:
        for el in elements:
            # Check if the element exists in the comma-separated list
            f = f[f['element_list'].str.contains(rf'\b{el}\b', regex=True, na=False)]

    f = f[(f["avg_hardness"].between(*hardness_range)) & (f["avg_density"].between(*density_range))]

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Hardness vs Density")
        fig = px.scatter(
            f, x="avg_hardness", y="avg_density",
            hover_data=["name", "formula", "crystal_system", "industry_application"],
            color="industry_application"
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Industry Distribution")
        vc = f["industry_application"].value_counts().reset_index()
        vc.columns = ["industry_application", "count"]
        fig2 = px.bar(vc, x="industry_application", y="count")
        st.plotly_chart(fig2, use_container_width=True)

    # Heatmap: Elements vs Industry
    st.subheader("Element vs Industry Heatmap")
    element_cols = [c for c in f.columns if c.startswith("element_") and c.endswith("_count")]
    if element_cols and len(f) > 0:
        heatmap_data = f.groupby("industry_application")[element_cols].mean()
        heatmap_data.columns = [c.replace("element_", "").replace("_count", "") for c in heatmap_data.columns]
        fig_heat = px.imshow(
            heatmap_data, 
            labels=dict(x="Element", y="Industry", color="Avg Count"),
            x=heatmap_data.columns,
            y=heatmap_data.index,
            aspect="auto",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("No data available for the heatmap.")

    st.subheader("Mineral Table")
    show_cols = ["id", "name", "formula", "crystal_system", "transparency_simple", "avg_hardness", "avg_density", "industry_application"]
    st.dataframe(f[show_cols], use_container_width=True, height=350)

    st.subheader("Predict Using Trained Models")
    if len(f) == 0:
        st.info("No rows after filtering. Adjust filters to select minerals.")
    else:
        pred_model = st.radio("Select Prediction Model Engine", ["Random Forest", "XGBoost"], horizontal=True)
        selected_idx = st.selectbox("Select a mineral row (by index in the filtered table)", f.index.tolist())
        row = df.loc[[selected_idx]]

        try:
            X_reg = row[num_features + cat_features_reg]
            X_clf = row[num_features + cat_features_clf]
            
            if pred_model == "Random Forest":
                pred_h = float(models["rf_reg_h"].predict(X_reg)[0])
                pred_d = float(models["rf_reg_d"].predict(X_reg)[0])
                pred_t = str(models["rf_clf_t"].predict(X_clf)[0])
            else:
                pred_h = float(models["xgb_reg_h"].predict(X_reg)[0])
                pred_d = float(models["xgb_reg_d"].predict(X_reg)[0])
                
                # XGBoost predicts numeric label, decode it
                pred_t_enc = models["xgb_clf_t"].predict(X_clf)[0]
                pred_t = str(models["xgb_le_t"].inverse_transform([pred_t_enc])[0])

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Predicted Avg Hardness", f"{pred_h:.2f}", delta=f"{pred_h - row['avg_hardness'].values[0]:.2f} (Error)")
            with m2:
                st.metric("Predicted Avg Density", f"{pred_d:.2f}", delta=f"{pred_d - row['avg_density'].values[0]:.2f} (Error)")
            with m3:
                actual_t = row['transparency_simple'].values[0]
                st.metric("Predicted Transparency", pred_t, delta="Match" if pred_t == actual_t else f"Actual: {actual_t}", delta_color="normal" if pred_t == actual_t else "inverse")

            st.caption(f"Predictions produced by {pred_model} trained on the cleaned dataset.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ==========================================
# TAB 2: CLUSTERING
# ==========================================
with tab_clustering:
    st.header("Unsupervised Learning: Mineral Clusters")
    
    if "cluster_kmeans" in df.columns:
        c1, c2 = st.columns([1, 3])
        with c1:
            color_option = st.selectbox(
                "Color Points By:",
                ["cluster_kmeans", "crystal_system", "industry_application", "transparency_simple"]
            )
            dim_red = st.radio("Dimensionality Reduction Method:", ["PCA", "t-SNE"])
            
        with c2:
            x_col = "pca_1" if dim_red == "PCA" else "tsne_1"
            y_col = "pca_2" if dim_red == "PCA" else "tsne_2"
            
            fig_clust = px.scatter(
                df, x=x_col, y=y_col,
                color=color_option,
                hover_data=["name", "formula", "cluster_kmeans", "crystal_system"],
                title=f"Mineral Groupings ({dim_red})"
            )
            fig_clust.update_traces(marker=dict(size=5, opacity=0.7))
            st.plotly_chart(fig_clust, use_container_width=True)
            
        st.subheader("Cluster Composition (K-Means)")
        if os.path.exists("outputs/reports/clustering/cluster_summary.csv"):
            c_summary = pd.read_csv("outputs/reports/clustering/cluster_summary.csv")
            st.dataframe(c_summary, use_container_width=True)
        else:
            st.info("Cluster summary CSV not found.")
    else:
        st.info("Clustering data not found in dataset. Please run the clustering script.")


# ==========================================
# TAB 3: EXPLORATORY DATA ANALYSIS
# ==========================================
with tab_eda:
    st.header("Exploratory Data Analysis")
    
    st.subheader("Feature Distributions")
    num_cols_for_dist = [c for c in df.columns if df[c].dtype in ['float64', 'int64'] and 'id' not in c.lower()]
    dist_col = st.selectbox("Select Numeric Feature for Distribution", sorted(num_cols_for_dist))
    if dist_col:
        fig_dist = px.histogram(df, x=dist_col, marginal="box", nbins=40, color_discrete_sequence=["#4C72B0"])
        st.plotly_chart(fig_dist, use_container_width=True)
    
    st.subheader("Class Balance")
    c1, c2, c3 = st.columns(3)
    with c1:
        fig_pie1 = px.pie(df, names="transparency_simple", title="Transparency")
        st.plotly_chart(fig_pie1, use_container_width=True)
    with c2:
        fig_pie2 = px.pie(df, names="crystal_system", title="Crystal System")
        st.plotly_chart(fig_pie2, use_container_width=True)
    with c3:
        fig_pie3 = px.pie(df, names="industry_application", title="Industry")
        st.plotly_chart(fig_pie3, use_container_width=True)
        
    st.subheader("Correlation Heatmap")
    corr_cols = [
        "avg_hardness", "avg_density", "mend_atomic_mass", "mend_electronegativity", 
        "rdkit_exact_mw", "rdkit_valence_electrons", "formula_weighted_mass", "formula_weighted_en"
    ]
    existing_corr_cols = [c for c in corr_cols if c in df.columns]
    if existing_corr_cols:
        corr_matrix = df[existing_corr_cols].corr()
        fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r")
        st.plotly_chart(fig_corr, use_container_width=True)


# ==========================================
# TAB 4: MODEL COMPARISON
# ==========================================
with tab_model_comp:
    st.header("Model Performance Comparison")
    
    if os.path.exists("outputs/reports/model_comparison.csv"):
        comp_df = pd.read_csv("outputs/reports/model_comparison.csv")
        
        st.subheader("Metrics Overview")
        st.dataframe(comp_df, use_container_width=True)
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Regression Performance (R²)")
            reg_df = comp_df[comp_df["task"].str.contains("regression")]
            fig_r2 = px.bar(reg_df, x="task", y="primary_metric", color="model", barmode="group", text_auto=".3f", labels={"primary_metric": "R²"})
            st.plotly_chart(fig_r2, use_container_width=True)
            
        with c2:
            st.subheader("Classification Performance (Accuracy)")
            clf_df = comp_df[~comp_df["task"].str.contains("regression")]
            fig_acc = px.bar(clf_df, x="task", y="primary_metric", color="model", barmode="group", text_auto=".3f", labels={"primary_metric": "Accuracy"})
            st.plotly_chart(fig_acc, use_container_width=True)
            
    else:
        st.info("Model comparison results not found. Please run the model comparison script.")


# ==========================================
# TAB 5: FEATURE IMPORTANCE
# ==========================================
with tab_feat_imp:
    st.header("Feature Importance")
    
    if os.path.exists("outputs/reports/feature_importance/all_importances.csv"):
        imp_df = pd.read_csv("outputs/reports/feature_importance/all_importances.csv")
        
        task_sel = st.selectbox("Select Task", imp_df["task"].unique())
        
        task_data = imp_df[imp_df["task"] == task_sel]
        
        c1, c2 = st.columns(2)
        with c1:
            rf_data = task_data[task_data["model"] == "RF"].head(15).sort_values("importance", ascending=True)
            fig_rf = px.bar(rf_data, x="importance", y="feature", orientation="h", title="Random Forest", color_discrete_sequence=["#4C72B0"])
            st.plotly_chart(fig_rf, use_container_width=True)
            
        with c2:
            xgb_data = task_data[task_data["model"] == "XGBoost"].head(15).sort_values("importance", ascending=True)
            fig_xgb = px.bar(xgb_data, x="importance", y="feature", orientation="h", title="XGBoost", color_discrete_sequence=["#DD8452"])
            st.plotly_chart(fig_xgb, use_container_width=True)
    else:
        st.info("Feature importance data not found. Please run the feature importance script.")