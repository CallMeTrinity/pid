import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from utils.data_loader import load_data

st.set_page_config(page_title="Exploration des données", page_icon="🔍", layout="wide")

st.title("Exploration des données")
st.markdown("Analyse descriptive du jeu de données cliniques (1 000 patients).")

df = load_data()

# ── Statistiques globales ────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Statistiques descriptives")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Variables quantitatives**")
    quant_cols = ["Age", "BMI", "Comorbidities", "Time_to_Event"]
    st.dataframe(df[quant_cols].describe().round(2), use_container_width=True)

with col2:
    st.markdown("**Répartition des variables catégorielles**")
    cat_summary = pd.DataFrame({
        "Variable": ["Sex", "Smoker", "Treatment", "Physical_Activity", "Event_Observed"],
        "Modalités": [
            df["Sex"].value_counts().to_dict(),
            {0: int((df["Smoker"] == 0).sum()), 1: int((df["Smoker"] == 1).sum())},
            df["Treatment"].value_counts().to_dict(),
            df["Physical_Activity"].value_counts().to_dict(),
            {0: int((df["Event_Observed"] == 0).sum()), 1: int((df["Event_Observed"] == 1).sum())},
        ],
    })
    st.dataframe(cat_summary, use_container_width=True)

# ── Distribution de Time_to_Event ────────────────────────────────────────────
st.markdown("---")
st.subheader("Distribution de la durée de suivi (Time_to_Event)")

col1, col2 = st.columns(2)
with col1:
    fig = px.histogram(
        df, x="Time_to_Event", nbins=40,
        title="Histogramme — Durée de suivi",
        labels={"Time_to_Event": "Temps (mois)"},
        color_discrete_sequence=["#1f77b4"],
    )
    median_val = df["Time_to_Event"].median()
    fig.add_vline(x=median_val, line_dash="dash", line_color="red",
                  annotation_text=f"Médiane: {median_val:.1f} mois")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.box(
        df, y="Time_to_Event",
        title="Boxplot — Durée de suivi",
        labels={"Time_to_Event": "Temps (mois)"},
        color_discrete_sequence=["#1f77b4"],
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Variables qualitatives ────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Distribution des variables qualitatives")

cat_cols = {
    "Sex": "Sexe",
    "Treatment": "Traitement",
    "Physical_Activity": "Activité physique",
}

cols = st.columns(3)
for col_widget, (col_name, col_label) in zip(cols, cat_cols.items()):
    counts = df[col_name].value_counts().reset_index()
    counts.columns = [col_label, "Effectif"]
    fig = px.bar(
        counts, x=col_label, y="Effectif",
        title=f"Répartition — {col_label}",
        color=col_label,
        color_discrete_sequence=px.colors.qualitative.Set2,
        text="Effectif",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False)
    col_widget.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)
for col_widget, (col_name, col_label) in zip([col1, col2], [("Smoker", "Fumeur"), ("Event_Observed", "Évènement observé")]):
    counts = df[col_name].value_counts().reset_index()
    counts.columns = [col_label, "Effectif"]
    counts[col_label] = counts[col_label].map(
        {0: "Non (0)", 1: "Oui (1)"} if col_name == "Smoker" else {0: "Censuré (0)", 1: "Décès (1)"}
    )
    fig = px.pie(
        counts, names=col_label, values="Effectif",
        title=f"Répartition — {col_label}",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    col_widget.plotly_chart(fig, use_container_width=True)

# ── Variables quantitatives ────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Distribution des variables quantitatives")

quant_vars = [
    ("Age", "Âge (années)"),
    ("BMI", "IMC (BMI)"),
    ("Comorbidités", "Nombre de comorbidités"),
]

cols = st.columns(3)
for col_widget, (col_name, col_label) in zip(cols, [("Age", "Âge (années)"), ("BMI", "IMC"), ("Comorbidities", "Comorbidités")]):
    fig = px.histogram(
        df, x=col_name, nbins=30,
        title=f"Distribution — {col_label}",
        labels={col_name: col_label},
        color_discrete_sequence=["#ff7f0e"],
    )
    mean_val = df[col_name].mean()
    median_val = df[col_name].median()
    fig.add_vline(x=mean_val, line_dash="dash", line_color="blue",
                  annotation_text=f"Moy: {mean_val:.1f}")
    fig.add_vline(x=median_val, line_dash="dot", line_color="red",
                  annotation_text=f"Méd: {median_val:.1f}")
    col_widget.plotly_chart(fig, use_container_width=True)

# ── Corrélation et exploration croisée ────────────────────────────────────────
st.markdown("---")
st.subheader("Exploration croisée")

col1, col2 = st.columns(2)
with col1:
    x_var = st.selectbox("Variable X", ["Age", "BMI", "Comorbidities", "Time_to_Event"], index=0)
with col2:
    color_var = st.selectbox("Colorier par", ["Sex", "Treatment", "Physical_Activity", "Smoker", "Event_Observed"])

fig = px.scatter(
    df, x=x_var, y="Time_to_Event",
    color=color_var.replace("_", " "),
    title=f"{x_var} vs Durée de suivi, par {color_var}",
    labels={"Time_to_Event": "Temps de suivi (mois)", x_var: x_var},
    color_discrete_sequence=px.colors.qualitative.Set1,
    opacity=0.6,
    color_discrete_map=None,
)
# remap color column properly
fig = px.scatter(
    df.assign(**{color_var: df[color_var].astype(str)}),
    x=x_var, y="Time_to_Event",
    color=color_var,
    title=f"{x_var} vs Durée de suivi, par {color_var}",
    labels={"Time_to_Event": "Temps de suivi (mois)"},
    opacity=0.6,
    color_discrete_sequence=px.colors.qualitative.Set1,
)
st.plotly_chart(fig, use_container_width=True)
