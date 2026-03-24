import streamlit as st
import pandas as pd
from utils.data_loader import load_data

st.set_page_config(page_title="Conclusions", page_icon="📋", layout="wide")

st.title("Synthèse et Conclusions")
st.markdown("Récapitulatif des principaux résultats de l'analyse de survie.")

df = load_data()

# ── Résultats KM ─────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Résultats Kaplan-Meier")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Survie médiane", "36.38 mois")
col2.metric("Survie à 1 an", "78.72%")
col3.metric("Survie à 2 ans", "60.93%")
col4.metric("Survie à 3 ans", "50.46%")
col5.metric("Survie à 5 ans", "33.33%")

# ── Résultats Cox ─────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Résultats du Modèle de Cox")

cox_results = pd.DataFrame([
    {
        "Variable": "Âge",
        "Hazard Ratio": 1.034,
        "IC 95%": "[1.026 – 1.042]",
        "p-value": "< 0.001",
        "Interprétation": "+3.4% de risque par année supplémentaire",
        "Significatif": "✅",
    },
    {
        "Variable": "Sexe (Femme)",
        "Hazard Ratio": 1.034,
        "IC 95%": "[0.891 – 1.200]",
        "p-value": "0.659",
        "Interprétation": "Pas de différence significative entre les sexes",
        "Significatif": "❌",
    },
    {
        "Variable": "Fumeur",
        "Hazard Ratio": 1.479,
        "IC 95%": "[1.267 – 1.727]",
        "p-value": "< 0.001",
        "Interprétation": "+47.9% de risque chez les fumeurs",
        "Significatif": "✅",
    },
    {
        "Variable": "Traitement Expérimental",
        "Hazard Ratio": 0.710,
        "IC 95%": "[0.600 – 0.840]",
        "p-value": "< 0.001",
        "Interprétation": "-29% de risque vs traitement standard",
        "Significatif": "✅",
    },
    {
        "Variable": "Activité Physique Haute",
        "Hazard Ratio": 0.480,
        "IC 95%": "[0.389 – 0.593]",
        "p-value": "< 0.001",
        "Interprétation": "-52% de risque vs activité faible",
        "Significatif": "✅",
    },
    {
        "Variable": "Activité Physique Modérée",
        "Hazard Ratio": 0.703,
        "IC 95%": "[0.591 – 0.835]",
        "p-value": "< 0.001",
        "Interprétation": "-29.7% de risque vs activité faible",
        "Significatif": "✅",
    },
])

st.dataframe(cox_results, use_container_width=True, hide_index=True)

# ── Facteurs de risque vs protecteurs ─────────────────────────────────────────
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Facteurs de risque (HR > 1)")
    st.error("""
**Âge élevé** (HR = 1.034, p < 0.001)
Chaque année supplémentaire augmente le risque de décès de 3.4%.

**Tabagisme** (HR = 1.479, p < 0.001)
Les fumeurs ont un risque de décès 47.9% plus élevé que les non-fumeurs.
""")

with col2:
    st.subheader("Facteurs protecteurs (HR < 1)")
    st.success("""
**Traitement expérimental** (HR = 0.710, p < 0.001)
Réduit le risque de décès de 29% par rapport au traitement standard.

**Activité physique haute** (HR = 0.480, p < 0.001)
Réduit le risque de décès de 52% par rapport à une activité faible.

**Activité physique modérée** (HR = 0.703, p < 0.001)
Réduit le risque de décès de 29.7% par rapport à une activité faible.
""")

# ── Insights cliniques ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Insights cliniques")

st.markdown("""
### Principales conclusions

1. **Le traitement expérimental est efficace** : il réduit significativement le risque de décès
   (-29%), ce qui suggère une supériorité clinique par rapport au traitement standard.

2. **L'activité physique est un facteur protecteur majeur** : une activité physique haute
   réduit le risque de décès de moitié (-52%). L'effet dose-réponse est visible
   (Low < Moderate < High).

3. **Le tabagisme est un facteur de risque important** : les fumeurs ont un risque de décès
   près de 48% plus élevé. Des programmes de sevrage tabagique pourraient améliorer le pronostic.

4. **L'âge impacte significativement le pronostic** : chaque année supplémentaire augmente
   le risque de 3.4%, soulignant l'importance d'une prise en charge précoce.

5. **Le sexe n'est pas un facteur pronostique significatif** (p = 0.659) : les hommes et
   les femmes ont des survies comparables dans cette cohorte.

### Limites de l'étude

- Données simulées (1 000 patients) — les résultats ne sont pas directement transposables
  à des situations cliniques réelles.
- L'hypothèse des risques proportionnels du modèle de Cox doit être vérifiée sur données réelles.
- Des variables confondantes non mesurées pourraient influencer les résultats.
""")

# ── Tableau de bord final ─────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Données de la cohorte")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Patients total", f"{len(df):,}")
col2.metric("Taux de décès", f"{df['Event_Observed'].mean()*100:.1f}%")
col3.metric("Taux de censure", f"{(1-df['Event_Observed'].mean())*100:.1f}%")
col4.metric("Âge moyen", f"{df['Age'].mean():.1f} ans")

st.markdown("---")
st.caption("Analyse réalisée dans le cadre du cours de Data Science — Master MIAGE M1 (2025-2026)")
