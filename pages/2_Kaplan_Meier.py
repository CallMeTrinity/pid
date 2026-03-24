import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from lifelines import KaplanMeierFitter
from utils.data_loader import load_data
from utils.plots import plot_km_global, plot_km_stratified, logrank_result, km_survival_table

st.set_page_config(page_title="Kaplan-Meier", page_icon="📈", layout="wide")

st.title("Estimateur de Kaplan-Meier")
st.markdown("""
L'estimateur de Kaplan-Meier est une méthode **non-paramétrique** qui estime la probabilité
de survie $S(t) = P(T > t)$ en tenant compte des données censurées.
""")

df = load_data()

# ── Courbe globale ────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Courbe de survie globale")

fig, kmf_global = plot_km_global(df)
st.pyplot(fig)
plt.close(fig)

# Tableau des probabilités à des temps clés
st.markdown("**Probabilités de survie estimées**")
time_points = [12, 24, 36, 60, 100]
table = km_survival_table(kmf_global, time_points)
st.dataframe(table, use_container_width=True, hide_index=True)

col1, col2, col3 = st.columns(3)
col1.metric("Survie médiane", f"{kmf_global.median_survival_time_:.2f} mois")
col2.metric("Survie à 1 an (12 mois)", f"{kmf_global.predict(12)*100:.2f}%")
col3.metric("Survie à 3 ans (36 mois)", f"{kmf_global.predict(36)*100:.2f}%")

# ── Analyse stratifiée ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Analyse stratifiée")

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

col1, col2 = st.columns([2, 1])

with col1:
    fig = plot_km_stratified(df, group_col)
    st.pyplot(fig)
    plt.close(fig)

with col2:
    st.markdown("**Test du Log-Rank**")
    result = logrank_result(df, group_col)
    p = result["p_value"]
    stat = result["statistic"]
    significance = "✅ Significatif (p < 0.05)" if p < 0.05 else "❌ Non significatif (p ≥ 0.05)"

    st.metric("Statistique de test", f"{stat:.4f}")
    st.metric("p-value", f"{p:.4f}")
    st.info(significance)

    if p < 0.05:
        st.success(
            f"Les courbes de survie diffèrent significativement selon **{group_label}** (p = {p:.4f})."
        )
    else:
        st.warning(
            f"Pas de différence significative de survie selon **{group_label}** (p = {p:.4f})."
        )

    st.markdown("**Survie médiane par groupe**")
    kmf = KaplanMeierFitter()
    medians = []
    for grp in sorted(df[group_col].dropna().unique(), key=str):
        mask = df[group_col] == grp
        kmf.fit(df.loc[mask, "Time_to_Event"], event_observed=df.loc[mask, "Event_Observed"])
        medians.append({"Groupe": str(grp), "Médiane (mois)": f"{kmf.median_survival_time_:.2f}"})
    st.dataframe(pd.DataFrame(medians), use_container_width=True, hide_index=True)

# ── Interprétation ────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("Interprétation — Kaplan-Meier"):
    st.markdown("""
**Comment lire une courbe de Kaplan-Meier ?**
- L'axe des **ordonnées** représente la probabilité de survie S(t) ∈ [0, 1]
- L'axe des **abscisses** représente le temps (ici en mois)
- Chaque **marche descendante** correspond à un évènement (décès)
- Les **intervalles de confiance** (zone ombrée) à 95% montrent l'incertitude de l'estimation
- Les **croix** sur la courbe indiquent les observations censurées

**Médiane de survie :** temps au-delà duquel 50% des patients ont survécu.

**Test du Log-Rank :**
- H₀ : les fonctions de survie sont identiques entre les groupes
- p < 0.05 → on rejette H₀, les groupes ont des survies significativement différentes
""")
