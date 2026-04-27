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
    "Treatment_Experimental": "Traitement Expérimental vs Standard",
    "Activity_High": "Activité Haute vs Basse",
    "Activity_Moderate": "Activité Modérée vs Basse",
}


def render(df: pd.DataFrame, time_col: str, event_col: str):
    st.subheader("Modèle de régression de Cox")
    st.markdown("""
    Le modèle de Cox permet d'analyser l'impact de plusieurs variables (âge, tabagisme, traitement, etc.) sur le risque de décès des patients au cours du temps.
    """)

    st.latex(r"h(t|X) = h_0(t) \cdot \exp(\beta_1 X_1 + \cdots + \beta_p X_p)")
    st.markdown("""
    Le **Hazard Ratio** $HR = e^{\\beta}$ quantifie l'effet de chaque covariable.

    Un Hazard Ratio (HR) supérieur à 1 indique une augmentation du risque de décès,
    tandis qu'un HR inférieur à 1 indique un effet protecteur. Si l'intervalle de
    confiance à 95% contient 1, l'effet n'est pas statistiquement significatif.""")

    # ── Fit model ─────────────────────────────────────────────────────────────
    cox_data = prepare_cox_data(df, time_col, event_col)
    h = hashlib.md5(cox_data.to_json().encode()).hexdigest()
    cph, dropped = fit_cox_model(h, cox_data, time_col, event_col)

    if dropped:
        st.info(f"Variables retirées (constantes après filtrage) : {', '.join(dropped)}")

    # ── Model summary ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Résultats du modèle")

    c_index = cph.concordance_index_
    col1, col2, col3 = st.columns(3)
    col1.metric("Concordance (C-index)", f"{c_index:.4f}")
    col2.metric("Observations", f"{int(cph.summary['z'].count() and len(cox_data))}")
    col3.metric("Événements", f"{int(cox_data[event_col].sum())}")

    if c_index >= 0.7:
        st.markdown(
            f"Le C-index de **{c_index:.3f}** indique une bonne capacité discriminante : "
            f"le modèle identifie correctement quel patient aura l'événement en premier dans "
            f"{c_index * 100:.0f}% des paires, contre 50% pour un modèle aléatoire."
        )
    elif c_index >= 0.6:
        st.markdown(
            f"Le C-index de **{c_index:.3f}** indique une capacité discriminante modérée : "
            f"le modèle classe correctement l'ordre des événements dans {c_index * 100:.0f}% "
            f"des paires : mieux que le hasard (50%), mais une valeur sous 0.7 suggère que "
            f"des variables supplémentaires ou des interactions pourraient améliorer la prédiction."
        )
    else:
        st.markdown(
            f"Le C-index de **{c_index:.3f}** est faible, proche du hasard (50%) : "
            f"les covariables incluses expliquent peu la variabilité des temps de survie. "
            f"L'ajout de nouvelles variables pourrait améliorer le modèle."
        )

    summary = cph.summary.copy().reset_index()
    summary.columns = [str(c) for c in summary.columns]

    display = summary[["covariate", "exp(coef)",
                       "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].copy()
    display.columns = ["Variable", "HR", "IC bas 95%", "IC haut 95%", "p-value"]
    display["Variable"] = display["Variable"].map(VAR_LABELS).fillna(display["Variable"])
    display["Significatif"] = display["p-value"].apply(lambda p: "Oui" if p < 0.05 else "Non")
    display["Effet"] = display.apply(
        lambda row: ("Risque (+)" if row["HR"] > 1 else "Protecteur (-)")
        if row["Significatif"] == "Oui" else "Non significatif",
        axis=1
    )

    for c in ["HR", "IC bas 95%", "IC haut 95%"]:
        display[c] = display[c].round(4)
    display["p-value"] = display["p-value"].apply(lambda p: f"{p:.6f}" if p >= 0.001 else "< 0.001")

    st.dataframe(display, use_container_width=True, hide_index=True)

    # ── Interpretation ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Interprétation des Hazard Ratios")

    # Preferred display order: risk factors first, sex grouped with demographics
    PREFERRED_ORDER = [
        "Smoker", "Age", "Activity_Moderate", "Activity_High",
        "Treatment_Experimental", "Sex_Female",
    ]

    def _sort_key(cov):
        return PREFERRED_ORDER.index(cov) if cov in PREFERRED_ORDER else 99

    risk_df = summary[summary["exp(coef)"] > 1].copy()
    risk_df["__order"] = risk_df["covariate"].apply(_sort_key)
    risk_df = risk_df.sort_values(["__order", "exp(coef)"], ascending=[True, False])

    prot_df = summary[summary["exp(coef)"] < 1].copy()
    prot_df["__order"] = prot_df["covariate"].apply(_sort_key)
    prot_df = prot_df.sort_values(["__order", "exp(coef)"], ascending=[True, True])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Facteurs de risque (HR > 1)**")
        for _, row in risk_df.iterrows():
            var = VAR_LABELS.get(row["covariate"], row["covariate"])
            hr = row["exp(coef)"]
            p = row["p"]
            pct = (hr - 1) * 100
            sig = " **(significatif)**" if p < 0.05 else " (non significatif)"
            st.markdown(f"- **{var}** : HR = {hr:.3f} → +{pct:.1f}% de risque{sig}")
    with col2:
        st.markdown("**Facteurs protecteurs (HR < 1)**")
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
        parts = []
        top_risk = sig_risk.sort_values("exp(coef)", ascending=False).iloc[0] if len(sig_risk) else None
        top_prot = sig_prot.sort_values("exp(coef)").iloc[0] if len(sig_prot) else None
        if top_risk is not None and top_prot is not None:
            r_name = VAR_LABELS.get(top_risk["covariate"], top_risk["covariate"])
            p_name = VAR_LABELS.get(top_prot["covariate"], top_prot["covariate"])
            r_hr = top_risk["exp(coef)"]
            p_hr = top_prot["exp(coef)"]
            r_pct = (r_hr - 1) * 100
            p_pct = (1 - p_hr) * 100
            parts.append(
                f"Globalement, les résultats montrent que **{r_name}** est le principal "
                f"facteur de risque de décès (HR = {r_hr:.3f}, soit +{r_pct:.0f}% de risque "
                f"de décès), tandis que **{p_name}** est le facteur le plus protecteur "
                f"sur la survie des patients (HR = {p_hr:.3f}, soit -{p_pct:.0f}% de risque)."
            )
        elif top_risk is not None:
            r_name = VAR_LABELS.get(top_risk["covariate"], top_risk["covariate"])
            r_pct = (top_risk["exp(coef)"] - 1) * 100
            parts.append(
                f"Globalement, **{r_name}** est le principal facteur de risque identifié "
                f"(HR = {top_risk['exp(coef)']:.3f}, soit +{r_pct:.0f}% de risque)."
            )
        elif top_prot is not None:
            p_name = VAR_LABELS.get(top_prot["covariate"], top_prot["covariate"])
            p_pct = (1 - top_prot["exp(coef)"]) * 100
            parts.append(
                f"Globalement, **{p_name}** est le facteur le plus protecteur identifié "
                f"(HR = {top_prot['exp(coef)']:.3f}, soit -{p_pct:.0f}% de risque)."
            )
        st.markdown(" ".join(parts))
    else:
        st.markdown(
            "Aucune covariable n'atteint le seuil de significativité (p < 0.05). "
            "Les effets observés pourraient être dus au hasard. Cela peut indiquer "
            "un manque de puissance statistique ou une absence réelle d'effet."
        )

    # ── Tableau d'impact des variables ────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Impact de chaque variable sur la survie")
    st.markdown("Ce tableau classe **toutes** les covariables du modèle (y compris les "
                "niveaux séparés de l'activité physique et du traitement) selon leur effet "
                "sur le risque de décès.")

    impact_df = summary[["covariate", "exp(coef)", "p"]].copy()
    impact_df["Variable"] = impact_df["covariate"].map(VAR_LABELS).fillna(impact_df["covariate"])
    impact_df["HR"] = impact_df["exp(coef)"].round(3)
    impact_df["Effet sur le risque"] = impact_df["exp(coef)"].apply(
        lambda hr: f"+{(hr-1)*100:.1f}%" if hr > 1 else f"-{(1-hr)*100:.1f}%"
    )
    impact_df["Direction"] = impact_df["exp(coef)"].apply(
        lambda hr: "Risque" if hr > 1 else "Protecteur"
    )
    impact_df["Significatif"] = impact_df["p"].apply(lambda p: "Oui" if p < 0.05 else "Non")
    impact_df["p-value"] = impact_df["p"].apply(
        lambda p: f"{p:.4f}" if p >= 0.001 else "< 0.001"
    )
    # Sort: significant first, then by effect magnitude
    impact_df["__mag"] = (impact_df["exp(coef)"] - 1).abs()
    impact_df["__sig_sort"] = (impact_df["p"] < 0.05).astype(int)
    impact_df = impact_df.sort_values(
        ["__sig_sort", "__mag"], ascending=[False, False]
    )
    st.dataframe(
        impact_df[["Variable", "HR", "Effet sur le risque",
                   "Direction", "Significatif", "p-value"]],
        use_container_width=True, hide_index=True,
    )

    # ── Forest plot ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Visualisation des Hazard Ratios")

    fig = plot_hazard_ratios(cph)
    st.pyplot(fig)
    plt.close(fig)

    # ── Adjusted survival curves ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Courbes de survie ajustées")

    st.markdown("Les courbes de survie ajustées permettent de visualiser l'impact "
    "réel d'une variable sur la survie en tenant compte simultanément des autres "
    "facteurs du modèle (âge, sexe, tabagisme, traitement, etc.). Elles permettent "
    "ainsi d'isoler l'impact réel d'une variable, indépendamment des autres "
    "caractéristiques des patients.\n\n"
    "Concrètement, cela signifie que deux patients ayant les mêmes caractéristiques, "
    "sauf pour une variable donnée (par exemple le tabagisme), n'auront pas le même "
    "risque de décès.")

    COV_OPTIONS = {
        "Traitement": ("Treatment_Experimental", [0, 1], ["Standard", "Experimental"]),
        "Fumeur": ("Smoker", [0, 1], ["Non", "Oui"]),
        "Sexe": ("Sex_Female", [0, 1], ["Homme", "Femme"]),
        "Activité Haute": ("Activity_High", [0, 1], ["Non", "Oui"]),
        "Activité Modérée": ("Activity_Moderate", [0, 1], ["Non", "Oui"]),
    }
    available = {k: v for k, v in COV_OPTIONS.items() if v[0] in cph.summary.index}

    if available:
        cov_label = st.selectbox("Covariable à comparer", list(available.keys()), key="cox_cov")
        covariate, values, value_labels = available[cov_label]

        fig, ax = plt.subplots(figsize=(10, 5))
        cph.plot_partial_effects_on_outcome(
            covariates=covariate, values=values,
            ax=ax, plot_baseline=False,
        )
        # Rename legend labels: 0 / 1 → meaningful
        handles, labels = ax.get_legend_handles_labels()
        new_labels = []
        for lab in labels:
            new = lab
            for v, disp in zip(values, value_labels):
                if lab.endswith(f"={v}") or lab == f"{covariate}={v}" or lab.strip() == str(v):
                    new = f"{cov_label} = {disp}"
                    break
            new_labels.append(new)
        ax.legend(handles, new_labels, title=cov_label)
        ax.set(title=f"Survie ajustée - {cov_label}", xlabel="Temps (mois)",
               ylabel="S(t)", ylim=(0, 1.05))
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Dynamic interpretation based on the Cox HR for this covariate
        try:
            hr = float(cph.summary.loc[covariate, "exp(coef)"])
            p_val = float(cph.summary.loc[covariate, "p"])
        except Exception:
            hr, p_val = None, None

        if hr is not None:
            # values[0] is the reference level, values[1] is the contrast
            ref_label = value_labels[0]
            cmp_label = value_labels[1]
            pct = abs(hr - 1) * 100
            sig_txt = "significatif" if p_val is not None and p_val < 0.05 else "non significatif"

            if hr < 1:
                effect_desc = (
                    f"une **meilleure survie** chez les patients du groupe *{cmp_label}* "
                    f"par rapport au groupe *{ref_label}* (HR = {hr:.3f}, soit -{pct:.0f}% "
                    f"de risque de décès, effet {sig_txt})"
                )
                confirm = "confirme son effet protecteur mis en évidence par le modèle de Cox."
            elif hr > 1:
                effect_desc = (
                    f"une **survie réduite** chez les patients du groupe *{cmp_label}* "
                    f"par rapport au groupe *{ref_label}* (HR = {hr:.3f}, soit +{pct:.0f}% "
                    f"de risque de décès, effet {sig_txt})"
                )
                confirm = "confirme son effet délétère mis en évidence par le modèle de Cox."
            else:
                effect_desc = f"un effet neutre (HR = {hr:.3f})"
                confirm = "suggère que la variable n'a pas d'impact marqué sur la survie."

            st.markdown(
                f"On observe {effect_desc}, ce qui {confirm}"
            )

    # ── Proportional hazards test ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Vérification de l'hypothèse des risques proportionnels")
    st.markdown("""
Le test des **résidus de Schoenfeld** vérifie que les HR sont constants dans le temps.
- H0 : les résidus ne dépendent pas du temps (hypothèse respectée)
- p > 0.05 → hypothèse validée pour cette covariable
""")

    try:
        from lifelines.statistics import proportional_hazard_test
        # Use only columns that the model was fitted on
        ph_cols = list(cph.summary.index) + [time_col, event_col]
        ph_result = proportional_hazard_test(cph, cox_data[ph_cols], time_transform="rank")
        ph_table = ph_result.summary.copy().reset_index()
        ph_table.columns = [str(c) for c in ph_table.columns]

        ph_display = ph_table.copy()
        if "-log2(p)" in ph_display.columns:
            ph_display = ph_display.drop(columns=["-log2(p)"])
        if ph_display.columns[0] != "Variable":
            ph_display = ph_display.rename(columns={ph_display.columns[0]: "Variable"})
        ph_display["Variable"] = ph_display["Variable"].map(VAR_LABELS).fillna(ph_display["Variable"])
        ph_display["Hypothèse respectée"] = ph_display["p"].apply(
            lambda p: "Oui" if p > 0.05 else "Non"
        )
        ph_display["p"] = ph_display["p"].apply(lambda p: f"{p:.4f}")
        st.dataframe(ph_display, use_container_width=True, hide_index=True)

        all_ok = all(ph_result.summary["p"] > 0.05)
        violated = [idx for idx, row in ph_result.summary.iterrows() if row["p"] <= 0.05]
        if all_ok:
            st.success("L'hypothèse des risques proportionnels est validée pour toutes les covariables.")
            st.markdown(
                "Cela signifie que l'effet des variables sur le risque de décès reste **constant dans le "
                "temps**. Le modèle de Cox est donc valide et ses résultats peuvent être interprétés de "
                "manière fiable."
            )
        else:
            violated_names = [VAR_LABELS.get(v, v) for v in violated]
            st.warning("L'hypothèse n'est pas respectée pour certaines covariables. "
                       "Les résultats du modèle de Cox doivent être interprétés avec prudence.")
            st.markdown(
                f"Les variables **{', '.join(violated_names)}** violent l'hypothèse de "
                f"proportionnalité : leur effet sur le risque **varie au cours du temps**. "
                f"Pour ces variables, le HR moyen rapporté ci-dessus ne reflète pas "
                f"fidèlement la réalité. Des modèles stratifiés ou à effets dépendants "
                f"du temps pourraient être plus adaptés."
            )
    except Exception as e:
        st.warning(f"Test non disponible : {e}")
