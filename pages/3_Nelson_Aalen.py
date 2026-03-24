import streamlit as st
import matplotlib.pyplot as plt
from lifelines import NelsonAalenFitter
from utils.data_loader import load_data
from utils.plots import plot_na_global, plot_na_stratified

st.set_page_config(page_title="Nelson-Aalen", page_icon="⚡", layout="wide")

st.title("Estimateur de Nelson-Aalen")
st.markdown("""
L'estimateur de Nelson-Aalen estime la **fonction de risque cumulée** :
$$H(t) = \\int_0^t h(u)\\,du$$
où $h(t)$ est le taux de risque instantané. Plus $H(t)$ croît rapidement,
plus le risque de décès est élevé à ce moment.
""")

df = load_data()

# ── Courbe globale ────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Risque cumulé global")

fig = plot_na_global(df)
st.pyplot(fig)
plt.close(fig)

naf = NelsonAalenFitter()
naf.fit(df["Time_to_Event"], event_observed=df["Event_Observed"])

col1, col2, col3 = st.columns(3)
col1.metric("H(t) à 12 mois", f"{naf.predict(12):.4f}")
col2.metric("H(t) à 36 mois", f"{naf.predict(36):.4f}")
col3.metric("H(t) à 60 mois", f"{naf.predict(60):.4f}")

st.markdown("""
> **Lecture :** Un risque cumulé de $H(t) = 1$ signifie que le risque accumulé
> équivaut à une probabilité de décès de 63% (en supposant une distribution exponentielle).
""")

# ── Analyse stratifiée ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Comparaison du risque cumulé par groupe")

GROUP_OPTIONS = {
    "Sexe": "Sex",
    "Traitement": "Treatment",
    "Activité physique": "Physical_Activity",
    "Fumeur": "Smoker",
    "Tranche d'âge": "Tranche_Age",
    "Tranche IMC": "Tranche_BMI",
}

group_label = st.selectbox(
    "Choisir la variable de stratification",
    options=list(GROUP_OPTIONS.keys()),
)
group_col = GROUP_OPTIONS[group_label]

fig = plot_na_stratified(df, group_col)
st.pyplot(fig)
plt.close(fig)

# Tableau comparatif H(t)
st.markdown("**Risque cumulé H(t) par groupe à des temps clés**")
time_points = [12, 24, 36, 60]
rows = []
naf = NelsonAalenFitter()
for grp in sorted(df[group_col].dropna().unique(), key=str):
    mask = df[group_col] == grp
    naf.fit(df.loc[mask, "Time_to_Event"], event_observed=df.loc[mask, "Event_Observed"])
    row = {"Groupe": str(grp)}
    for t in time_points:
        row[f"H({t} mois)"] = f"{naf.predict(t):.4f}"
    rows.append(row)

import pandas as pd
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── Relation KM / NA ──────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("Relation entre Kaplan-Meier et Nelson-Aalen"):
    st.markdown("""
Les deux estimateurs sont liés par la relation :

$$S(t) \\approx e^{-H(t)}$$

- **Kaplan-Meier** estime directement $S(t)$ (probabilité de survie)
- **Nelson-Aalen** estime $H(t)$ (risque cumulé), plus stable pour les petits échantillons

**Interprétation de la pente de H(t) :**
- Pente **forte** → taux de risque instantané élevé (beaucoup de décès dans cet intervalle)
- Pente **faible** → taux de risque faible (peu de décès dans cet intervalle)
- Courbe **linéaire** → risque constant dans le temps (distribution exponentielle)
""")
