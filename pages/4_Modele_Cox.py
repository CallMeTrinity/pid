import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils.data_loader import load_data, prepare_cox_data, fit_cox_model
from utils.plots import plot_hazard_ratios

st.set_page_config(page_title="Modèle de Cox", page_icon="🔬", layout="wide")

st.title("Modèle de Cox à Risques Proportionnels")
st.markdown("""
Le modèle de Cox est un modèle **semi-paramétrique** qui modélise le taux de risque :
$$h(t|X) = h_0(t) \\cdot e^{\\beta_1 X_1 + \\beta_2 X_2 + \\cdots + \\beta_p X_p}$$

Le **Hazard Ratio** (HR = $e^{\\beta}$) mesure l'effet d'une covariable sur le risque :
- HR > 1 → facteur de risque (augmente le risque de décès)
- HR < 1 → facteur protecteur (réduit le risque de décès)
""")

df = load_data()

with st.spinner("Ajustement du modèle de Cox..."):
    cph = fit_cox_model()

# ── Résumé du modèle ─────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Résultats du modèle")

summary = cph.summary.copy()
summary = summary.reset_index()
summary.columns = [str(c) for c in summary.columns]

# Renommer les variables pour l'affichage
VAR_LABELS = {
    "Age": "Âge",
    "Sex_Female": "Sexe (Femme vs Homme)",
    "Smoker": "Fumeur (Oui vs Non)",
    "Treatment_Experimental": "Traitement Expérimental vs Standard",
    "Activity_High": "Activité Haute vs Basse",
    "Activity_Moderate": "Activité Modérée vs Basse",
}

display_summary = summary[["covariate", "exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].copy()
display_summary.columns = ["Variable", "Hazard Ratio", "IC 95% bas", "IC 95% haut", "p-value"]
display_summary["Variable"] = display_summary["Variable"].map(VAR_LABELS).fillna(display_summary["Variable"])
display_summary["Significatif"] = display_summary["p-value"].apply(lambda p: "✅" if p < 0.05 else "❌")
display_summary["Effet"] = display_summary["Hazard Ratio"].apply(
    lambda hr: "Risque (+)" if hr > 1 else "Protecteur (-)"
)

for col in ["Hazard Ratio", "IC 95% bas", "IC 95% haut"]:
    display_summary[col] = display_summary[col].round(3)
display_summary["p-value"] = display_summary["p-value"].apply(lambda p: f"{p:.4f}")

st.dataframe(display_summary, use_container_width=True, hide_index=True)

col1, col2 = st.columns([1, 1])
with col1:
    st.metric("Concordance (C-index)", f"{cph.concordance_index_:.4f}")
with col2:
    st.info("Un C-index > 0.5 indique un modèle meilleur que le hasard. C-index = 1 indique un modèle parfait.")

# ── Forest plot des HR ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Visualisation des Hazard Ratios")

fig = plot_hazard_ratios(cph)
st.pyplot(fig)
plt.close(fig)

# ── Courbes de survie ajustées ────────────────────────────────────────────────
st.markdown("---")
st.subheader("Courbes de survie ajustées")

st.markdown("Comparaison de la survie pour différentes valeurs d'une covariable (les autres variables étant à leur moyenne).")

COVARIATE_OPTIONS = {
    "Traitement (Standard=0 / Expérimental=1)": ("Treatment_Experimental", [0, 1]),
    "Fumeur (Non=0 / Oui=1)": ("Smoker", [0, 1]),
    "Sexe (Homme=0 / Femme=1)": ("Sex_Female", [0, 1]),
    "Activité Haute (Non=0 / Oui=1)": ("Activity_High", [0, 1]),
    "Activité Modérée (Non=0 / Oui=1)": ("Activity_Moderate", [0, 1]),
}

covariate_label = st.selectbox("Choisir la covariable à comparer", list(COVARIATE_OPTIONS.keys()))
covariate, values = COVARIATE_OPTIONS[covariate_label]

fig, ax = plt.subplots(figsize=(10, 5))
try:
    cph.plot_partial_effects_on_outcome(
        covariates=covariate,
        values=values,
        ax=ax,
        plot_baseline=False,
    )
    ax.set_title(f"Courbes de survie ajustées — {covariate_label}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Temps (mois)")
    ax.set_ylabel("Probabilité de survie S(t)")
    ax.set_ylim(0, 1.05)
    st.pyplot(fig)
except Exception as e:
    st.error(f"Erreur lors du tracé : {e}")
plt.close(fig)

# ── Prédiction individuelle ───────────────────────────────────────────────────
st.markdown("---")
st.subheader("Prédiction de survie pour un profil individuel")

st.markdown("Renseignez les caractéristiques d'un patient pour obtenir sa courbe de survie prédite.")

col1, col2, col3 = st.columns(3)
with col1:
    age_input = st.slider("Âge", min_value=30, max_value=90, value=60)
    sex_input = st.selectbox("Sexe", ["Homme", "Femme"])
with col2:
    smoker_input = st.selectbox("Fumeur", ["Non", "Oui"])
    treatment_input = st.selectbox("Traitement", ["Standard", "Expérimental"])
with col3:
    activity_input = st.selectbox("Activité physique", ["Low", "Moderate", "High"])

sex_val = 1 if sex_input == "Femme" else 0
smoker_val = 1 if smoker_input == "Oui" else 0
treatment_val = 1 if treatment_input == "Expérimental" else 0
activity_high = 1 if activity_input == "High" else 0
activity_moderate = 1 if activity_input == "Moderate" else 0

patient_profile = pd.DataFrame([{
    "Age": age_input,
    "Sex_Female": sex_val,
    "Smoker": smoker_val,
    "Treatment_Experimental": treatment_val,
    "Activity_High": activity_high,
    "Activity_Moderate": activity_moderate,
}])

if st.button("Calculer la courbe de survie"):
    fig, ax = plt.subplots(figsize=(10, 5))
    try:
        surv = cph.predict_survival_function(patient_profile)
        ax.plot(surv.index, surv.iloc[:, 0], color="#1f77b4", linewidth=2, label="Profil patient")
        ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="S(t) = 0.5")
        ax.set_title("Courbe de survie prédite pour ce profil", fontsize=14, fontweight="bold")
        ax.set_xlabel("Temps (mois)")
        ax.set_ylabel("Probabilité de survie S(t)")
        ax.set_ylim(0, 1.05)
        ax.legend()
        st.pyplot(fig)

        # Probabilités aux temps clés
        time_points = [12, 24, 36, 60]
        cols = st.columns(len(time_points))
        for col_w, t in zip(cols, time_points):
            prob_at_t = float(surv[surv.index <= t].iloc[-1, 0]) if (surv.index <= t).any() else float(surv.iloc[0, 0])
            col_w.metric(f"Survie à {t} mois", f"{prob_at_t*100:.1f}%")
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
    plt.close(fig)

# ── Vérification des hypothèses ───────────────────────────────────────────────
st.markdown("---")
with st.expander("Vérification de l'hypothèse des risques proportionnels"):
    st.markdown("""
Le modèle de Cox suppose que le **rapport des risques (HR) est constant dans le temps**.

Cette hypothèse peut être vérifiée avec le **test de Schoenfeld** :
- H₀ : les résidus de Schoenfeld ne sont pas corrélés au temps (hypothèse vérifiée)
- p > 0.05 → l'hypothèse est satisfaite pour cette covariable

**Résultats du test (calculé lors de l'ajustement) :**
""")
    try:
        cox_data = prepare_cox_data(df)
        from lifelines.statistics import proportional_hazard_test
        results_ph = proportional_hazard_test(cph, cox_data, time_transform="rank")
        ph_table = results_ph.summary.copy().reset_index()
        ph_table.columns = [str(c) for c in ph_table.columns]
        ph_table["Variable"] = ph_table.iloc[:, 0].map(VAR_LABELS).fillna(ph_table.iloc[:, 0])
        ph_table["Hypothèse vérifiée"] = ph_table["p"].apply(lambda p: "✅ Oui" if p > 0.05 else "❌ Non")
        ph_table["p"] = ph_table["p"].apply(lambda p: f"{p:.4f}")
        st.dataframe(ph_table[["Variable", "test_statistic", "p", "Hypothèse vérifiée"]], use_container_width=True, hide_index=True)
    except Exception as e:
        st.warning(f"Test non disponible : {e}")
