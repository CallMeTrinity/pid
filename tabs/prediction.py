import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from utils.data_loader import prepare_cox_data, fit_cox_model
import hashlib


def render(df: pd.DataFrame, time_col: str, event_col: str):
    st.subheader("Prediction de survie d'un individu")
    st.markdown("""
A partir du modele de Cox ajuste sur les donnees, saisissez les
caracteristiques d'un patient pour visualiser sa courbe de survie predite.
""")

    # Fit model on full data (cached)
    cox_data = prepare_cox_data(df, time_col, event_col)
    h = hashlib.md5(cox_data.to_json().encode()).hexdigest()
    cph, dropped = fit_cox_model(h, cox_data, time_col, event_col)

    # ── Patient profile input ─────────────────────────────────────────────────
    st.markdown("#### Profil du patient")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 30, 90, 60, key="pred_age")
        sex = st.selectbox("Sexe", ["Homme", "Femme"], key="pred_sex")
    with col2:
        smoker = st.selectbox("Fumeur", ["Non", "Oui"], key="pred_smoker")
        treatment = st.selectbox("Traitement", ["Standard", "Experimental"], key="pred_treat")
    with col3:
        activity = st.selectbox("Activite physique", ["Low", "Moderate", "High"], key="pred_act")

    profile = pd.DataFrame([{
        "Age": age,
        "Sex_Female": 1 if sex == "Femme" else 0,
        "Smoker": 1 if smoker == "Oui" else 0,
        "Treatment_Experimental": 1 if treatment == "Experimental" else 0,
        "Activity_High": 1 if activity == "High" else 0,
        "Activity_Moderate": 1 if activity == "Moderate" else 0,
    }])

    # Keep only columns present in model
    model_cols = [c for c in cph.summary.index if c in profile.columns]
    profile = profile[model_cols]

    # ── Prediction ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Courbe de survie predite")

    surv_func = cph.predict_survival_function(profile)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(surv_func.index, surv_func.iloc[:, 0], color="#1f77b4", lw=2, label="Profil patient")
    ax.fill_between(surv_func.index, surv_func.iloc[:, 0], alpha=.15, color="#1f77b4")
    ax.axhline(y=.5, color="red", ls="--", alpha=.5, label="S(t) = 0.5")
    ax.set(
        title="Courbe de survie predite (Modele de Cox)",
        xlabel="Temps (mois)", ylabel="S(t)",
        ylim=(0, 1.05),
    )
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Key time points
    st.markdown("#### Probabilites de survie estimees")
    time_points = [12, 24, 36, 60, 100]
    cols = st.columns(len(time_points))
    for col_w, t in zip(cols, time_points):
        if (surv_func.index <= t).any():
            prob = float(surv_func[surv_func.index <= t].iloc[-1, 0])
        else:
            prob = 1.0
        col_w.metric(f"S({t} mois)", f"{prob*100:.1f}%")

    # ── Comparison with reference profiles ────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Comparaison avec des profils de reference")
    st.markdown("""
    - **Profil haut risque** : Homme, 70 ans, fumeur, traitement standard, activite faible
    - **Profil intermediaire** : Homme, 55 ans, non-fumeur, traitement standard, activite moderee
    - **Profil protege** : Femme, 45 ans, non-fumeuse, traitement experimental, activite haute
    """)

    ref_profiles = pd.DataFrame([
        {"Age": 70, "Sex_Female": 0, "Smoker": 1, "Treatment_Experimental": 0, "Activity_High": 0, "Activity_Moderate": 0},
        {"Age": 55, "Sex_Female": 0, "Smoker": 0, "Treatment_Experimental": 0, "Activity_High": 0, "Activity_Moderate": 1},
        {"Age": 45, "Sex_Female": 1, "Smoker": 0, "Treatment_Experimental": 1, "Activity_High": 1, "Activity_Moderate": 0},
    ])
    ref_profiles = ref_profiles[model_cols]
    labels = ["Haut risque", "Intermediaire", "Protege"]
    colors = ["#d62728", "#ff7f0e", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(10, 5))
    # Current patient
    ax.plot(surv_func.index, surv_func.iloc[:, 0], color="#1f77b4", lw=2.5, label="Votre profil")

    for i, (_, row) in enumerate(ref_profiles.iterrows()):
        sf = cph.predict_survival_function(row.to_frame().T)
        ax.plot(sf.index, sf.iloc[:, 0], color=colors[i], lw=1.5, ls="--", label=labels[i])

    ax.axhline(y=.5, color="grey", ls=":", alpha=.5)
    ax.set(title="Comparaison des courbes de survie",
           xlabel="Temps (mois)", ylabel="S(t)", ylim=(0, 1.05))
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
