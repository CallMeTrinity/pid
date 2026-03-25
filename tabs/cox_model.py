import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import hashlib
from utils.data_loader import prepare_cox_data, fit_cox_model
from utils.plots import plot_hazard_ratios


VAR_LABELS = {
    "Age": "Age",
    "Sex_Female": "Sexe (Femme vs Homme)",
    "Smoker": "Fumeur (Oui vs Non)",
    "Treatment_Experimental": "Traitement Experimental vs Standard",
    "Activity_High": "Activite Haute vs Basse",
    "Activity_Moderate": "Activite Moderee vs Basse",
}


def render(df: pd.DataFrame, time_col: str, event_col: str):
    st.subheader("Modele de regression de Cox")
    st.markdown("""
Le modele de Cox a risques proportionnels modelise le taux de risque :

$$h(t|X) = h_0(t) \\cdot \\exp(\\beta_1 X_1 + \\cdots + \\beta_p X_p)$$

Le **Hazard Ratio** $HR = e^{\\beta}$ quantifie l'effet de chaque covariable.
""")

    # ── Fit model ─────────────────────────────────────────────────────────────
    cox_data = prepare_cox_data(df, time_col, event_col)
    h = hashlib.md5(cox_data.to_json().encode()).hexdigest()
    cph, dropped = fit_cox_model(h, cox_data, time_col, event_col)

    if dropped:
        st.info(f"Variables retirees (constantes apres filtrage) : {', '.join(dropped)}")

    # ── Model summary ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Resultats du modele")

    c_index = cph.concordance_index_
    col1, col2, col3 = st.columns(3)
    col1.metric("Concordance (C-index)", f"{c_index:.4f}")
    col2.metric("Observations", f"{int(cph.summary['z'].count() and len(cox_data))}")
    col3.metric("Evenements", f"{int(cox_data[event_col].sum())}")

    if c_index >= 0.7:
        st.markdown(
            f"Le C-index de **{c_index:.3f}** indique une bonne capacite discriminante du modele "
            f"(un C-index de 0.5 correspond au hasard, 1.0 a une discrimination parfaite). "
            f"Le modele distingue correctement les patients a haut et bas risque dans environ "
            f"{c_index*100:.0f}% des paires de patients."
        )
    elif c_index >= 0.6:
        st.markdown(
            f"Le C-index de **{c_index:.3f}** indique une capacite discriminante moderee. "
            f"Le modele fait mieux que le hasard (0.5) mais pourrait etre ameliore en "
            f"integrant d'autres variables ou des interactions."
        )
    else:
        st.markdown(
            f"Le C-index de **{c_index:.3f}** est faible, proche du hasard. "
            f"Les covariables incluses dans le modele n'expliquent qu'une faible "
            f"part de la variabilite de la survie."
        )

    summary = cph.summary.copy().reset_index()
    summary.columns = [str(c) for c in summary.columns]

    display = summary[["covariate", "coef", "exp(coef)",
                       "exp(coef) lower 95%", "exp(coef) upper 95%", "z", "p"]].copy()
    display.columns = ["Variable", "Coef (beta)", "HR", "IC bas 95%", "IC haut 95%", "z", "p-value"]
    display["Variable"] = display["Variable"].map(VAR_LABELS).fillna(display["Variable"])
    display["Significatif"] = display["p-value"].apply(lambda p: "Oui" if p < 0.05 else "Non")
    display["Effet"] = display["HR"].apply(lambda hr: "Risque (+)" if hr > 1 else "Protecteur (-)")

    for c in ["Coef (beta)", "HR", "IC bas 95%", "IC haut 95%", "z"]:
        display[c] = display[c].round(4)
    display["p-value"] = display["p-value"].apply(lambda p: f"{p:.6f}" if p >= 0.001 else "< 0.001")

    st.dataframe(display, use_container_width=True, hide_index=True)

    # ── Interpretation ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Interpretation des Hazard Ratios")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Facteurs de risque (HR > 1)**")
        risk_df = summary[summary["exp(coef)"] > 1].copy()
        for _, row in risk_df.iterrows():
            var = VAR_LABELS.get(row["covariate"], row["covariate"])
            hr = row["exp(coef)"]
            p = row["p"]
            pct = (hr - 1) * 100
            sig = " **(significatif)**" if p < 0.05 else " (non significatif)"
            st.markdown(f"- **{var}** : HR = {hr:.3f} → +{pct:.1f}% de risque{sig}")
    with col2:
        st.markdown("**Facteurs protecteurs (HR < 1)**")
        prot_df = summary[summary["exp(coef)"] < 1].copy()
        for _, row in prot_df.iterrows():
            var = VAR_LABELS.get(row["covariate"], row["covariate"])
            hr = row["exp(coef)"]
            p = row["p"]
            pct = (1 - hr) * 100
            sig = " **(significatif)**" if p < 0.05 else " (non significatif)"
            st.markdown(f"- **{var}** : HR = {hr:.3f} → -{pct:.1f}% de risque{sig}")

    # ── Global interpretation ────────────────────────────────────────────────
    sig_risk = summary[(summary["exp(coef)"] > 1) & (summary["p"] < 0.05)]
    sig_prot = summary[(summary["exp(coef)"] < 1) & (summary["p"] < 0.05)]

    if len(sig_risk) > 0 or len(sig_prot) > 0:
        interp_parts = []
        if len(sig_risk) > 0:
            top_risk = sig_risk.sort_values("exp(coef)", ascending=False).iloc[0]
            risk_name = VAR_LABELS.get(top_risk["covariate"], top_risk["covariate"])
            risk_pct = (top_risk["exp(coef)"] - 1) * 100
            interp_parts.append(
                f"Le facteur de risque le plus important est **{risk_name}** "
                f"(HR = {top_risk['exp(coef)']:.3f}, soit +{risk_pct:.0f}% de risque de deces)."
            )
        if len(sig_prot) > 0:
            top_prot = sig_prot.sort_values("exp(coef)").iloc[0]
            prot_name = VAR_LABELS.get(top_prot["covariate"], top_prot["covariate"])
            prot_pct = (1 - top_prot["exp(coef)"]) * 100
            interp_parts.append(
                f"Le facteur protecteur le plus fort est **{prot_name}** "
                f"(HR = {top_prot['exp(coef)']:.3f}, soit -{prot_pct:.0f}% de risque)."
            )
        st.markdown(" ".join(interp_parts))
    else:
        st.markdown(
            "Aucune covariable n'atteint le seuil de significativite (p < 0.05). "
            "Les effets observes pourraient etre dus au hasard. Cela peut indiquer "
            "un manque de puissance statistique ou une absence reelle d'effet."
        )

    # ── Forest plot ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Visualisation des Hazard Ratios")

    fig = plot_hazard_ratios(cph)
    st.pyplot(fig)
    plt.close(fig)

    # ── Adjusted survival curves ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Courbes de survie ajustees")

    COV_OPTIONS = {
        "Traitement (Standard=0 / Experimental=1)": ("Treatment_Experimental", [0, 1]),
        "Fumeur (Non=0 / Oui=1)": ("Smoker", [0, 1]),
        "Sexe (Homme=0 / Femme=1)": ("Sex_Female", [0, 1]),
        "Activite Haute (Non=0 / Oui=1)": ("Activity_High", [0, 1]),
        "Activite Moderee (Non=0 / Oui=1)": ("Activity_Moderate", [0, 1]),
    }
    available = {k: v for k, v in COV_OPTIONS.items() if v[0] in cph.summary.index}

    if available:
        cov_label = st.selectbox("Covariable a comparer", list(available.keys()), key="cox_cov")
        covariate, values = available[cov_label]

        fig, ax = plt.subplots(figsize=(10, 5))
        cph.plot_partial_effects_on_outcome(
            covariates=covariate, values=values,
            ax=ax, plot_baseline=False,
        )
        ax.set(title=f"Survie ajustee — {cov_label}", xlabel="Temps (mois)",
               ylabel="S(t)", ylim=(0, 1.05))
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Proportional hazards test ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Verification de l'hypothese des risques proportionnels")
    st.markdown("""
Le test des **residus de Schoenfeld** verifie que les HR sont constants dans le temps.
- H0 : les residus ne dependent pas du temps (hypothese respectee)
- p > 0.05 → hypothese validee pour cette covariable
""")

    try:
        from lifelines.statistics import proportional_hazard_test
        # Use only columns that the model was fitted on
        ph_cols = list(cph.summary.index) + [time_col, event_col]
        ph_result = proportional_hazard_test(cph, cox_data[ph_cols], time_transform="rank")
        ph_table = ph_result.summary.copy().reset_index()
        ph_table.columns = [str(c) for c in ph_table.columns]

        ph_display = ph_table.copy()
        if ph_display.columns[0] != "Variable":
            ph_display = ph_display.rename(columns={ph_display.columns[0]: "Variable"})
        ph_display["Variable"] = ph_display["Variable"].map(VAR_LABELS).fillna(ph_display["Variable"])
        ph_display["Hypothese respectee"] = ph_display["p"].apply(
            lambda p: "Oui" if p > 0.05 else "Non"
        )
        ph_display["p"] = ph_display["p"].apply(lambda p: f"{p:.4f}")
        st.dataframe(ph_display, use_container_width=True, hide_index=True)

        all_ok = all(ph_result.summary["p"] > 0.05)
        violated = [idx for idx, row in ph_result.summary.iterrows() if row["p"] <= 0.05]
        if all_ok:
            st.success("L'hypothese des risques proportionnels est validee pour toutes les covariables.")
            st.markdown(
                "Cela signifie que l'effet de chaque variable sur le risque est **constant "
                "dans le temps**. Les Hazard Ratios estimes ci-dessus sont valables sur "
                "toute la duree du suivi."
            )
        else:
            violated_names = [VAR_LABELS.get(v, v) for v in violated]
            st.warning("L'hypothese n'est pas respectee pour certaines covariables. "
                       "Les resultats du modele de Cox doivent etre interpretes avec prudence.")
            st.markdown(
                f"Les variables **{', '.join(violated_names)}** violent l'hypothese de "
                f"proportionnalite : leur effet sur le risque **varie au cours du temps**. "
                f"Pour ces variables, le HR moyen rapporte ci-dessus ne reflete pas "
                f"fidelement la realite. Des modeles stratifies ou a effets dependants "
                f"du temps pourraient etre plus adaptes."
            )
    except Exception as e:
        st.warning(f"Test non disponible : {e}")
