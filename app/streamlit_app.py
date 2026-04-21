import pandas as pd
import streamlit as st
import plotly.express as px
import joblib

st.set_page_config(page_title="Mineral ML Explorer", layout="wide")

df = pd.read_csv("outputs/reports/cleaned.csv")

reg_h = joblib.load("outputs/models/rf_reg_avg_hardness.pkl")
reg_d = joblib.load("outputs/models/rf_reg_avg_density.pkl")
clf_t = joblib.load("outputs/models/rf_clf_transparency_simple.pkl")

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
cat_features_reg = ["crystal_system", "cleavage", "transparency", "industry_application"]
cat_features_clf = ["crystal_system", "cleavage", "industry_application"]

st.title("Mineral ML Explorer Dashboard")

with st.sidebar:
    st.header("Filters")

    industry = st.multiselect("Industry", sorted(df["industry_application"].astype(str).unique()))
    crystal = st.multiselect("Crystal system", sorted(df["crystal_system"].astype(str).unique()))

    h_min, h_max = float(df["avg_hardness"].min()), float(df["avg_hardness"].max())
    d_min, d_max = float(df["avg_density"].min()), float(df["avg_density"].max())

    hardness_range = st.slider("Avg hardness", h_min, h_max, (h_min, h_max))
    density_range = st.slider("Avg density", d_min, d_max, (d_min, d_max))

f = df.copy()
if industry:
    f = f[f["industry_application"].astype(str).isin(industry)]
if crystal:
    f = f[f["crystal_system"].astype(str).isin(crystal)]

f = f[(f["avg_hardness"].between(*hardness_range)) & (f["avg_density"].between(*density_range))]

c1, c2 = st.columns(2)

with c1:
    st.subheader("Hardness vs Density")
    fig = px.scatter(
        f,
        x="avg_hardness",
        y="avg_density",
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

st.subheader("Mineral Table")
show_cols = ["id", "name", "formula", "crystal_system", "transparency", "avg_hardness", "avg_density", "industry_application"]
st.dataframe(f[show_cols], use_container_width=True, height=350)

st.subheader("Predict Using Trained Models")
if len(f) == 0:
    st.info("No rows after filtering. Adjust filters to select minerals.")
else:
    selected_idx = st.selectbox(
        "Select a mineral row (by index in the filtered table)",
        f.index.tolist()
    )
    row = df.loc[[selected_idx]]

    X_reg = row[num_features + cat_features_reg]
    pred_h = float(reg_h.predict(X_reg)[0])
    pred_d = float(reg_d.predict(X_reg)[0])

    X_clf = row[num_features + cat_features_clf]
    pred_t = str(clf_t.predict(X_clf)[0])

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Predicted Avg Hardness", f"{pred_h:.2f}")
    with m2:
        st.metric("Predicted Avg Density", f"{pred_d:.2f}")
    with m3:
        st.metric("Predicted Transparency", pred_t)

    st.caption("Predictions are produced by the Random Forest models trained on the cleaned dataset.")