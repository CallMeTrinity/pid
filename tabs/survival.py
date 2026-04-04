import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, NelsonAalenFitter
from utils.plots import (
    plot_km_global, plot_km_stratified,
    km_survival_table, km_median_by_group, logrank_result,
    plot_na_global, plot_na_stratified,
)


GROUP_OPTIONS = {
    "Sexe": "Sex",
    "Traitement": "Treatment",
    "Activité physique": "Physical_Activity",
    "Fumeur": "Smoker",
    "Tranche d'âge": "Tranche_Age",
    "Tranche IMC": "Tranche_BMI",
}


def render(df: pd.DataFrame, time_col: str, event_col: str):
    st.subheader("Probabilités de survie et courbes de survie")

    available_groups = {k: v for k, v in GROUP_OPTIONS.items() if v in df.columns}

    section = st.radio(
        "Section", ["Kaplan-Meier", "Nelson-Aalen"],
        horizontal=True, key="surv_section",
    )

    # ══════════════════════════════════════════════════════════════════════════
    # KAPLAN-MEIER
    # ══════════════════════════════════════════════════════════════════════════
    if section == "Kaplan-Meier":
        st.markdown("---")
        st.markdown("### Estimateur de Kaplan-Meier")
        st.markdown("L’estimateur de Kaplan-Meier permet d’estimer la probabilité de survie des patients au cours du temps, en tenant compte du fait que certains patients peuvent être censurés (perdus de vue ou encore en vie à la fin de l’étude).")
        st.latex(r"\hat{S}(t) = \prod_{t_i \le t} \left(1 - \frac{d_i}{n_i}\right)")

        # Global curve
        st.markdown("#### Courbe de survie globale")
        st.markdown("La courbe de Kaplan-Meier montre l’évolution de la probabilité de survie des patients au cours du temps.")
        fig, kmf_global = plot_km_global(df, time_col, event_col)
        st.pyplot(fig)
        plt.close(fig)

        col1, col2, col3 = st.columns(3)
        median_surv = kmf_global.median_survival_time_
        s12 = float(kmf_global.predict(12))
        s36 = float(kmf_global.predict(36))
        col1.metric("Survie mediane", f"{median_surv:.2f} mois")
        col2.metric("Survie a 12 mois", f"{s12*100:.2f}%")
        col3.metric("Survie a 36 mois", f"{s36*100:.2f}%")

        # Interpretation
        st.markdown(
            f"La survie mediane est estimee a **{median_surv:.1f} mois**, ce qui signifie "
            f"que 50% des patients survivent au-dela de cette duree. "
            f"A 12 mois, **{s12*100:.1f}%** des patients sont encore en vie ; "
            f"a 36 mois, cette proportion descend a **{s36*100:.1f}%**. Ce qui traduit une diminution progressive de la probabilité de survie au cours du temps."
        )
        if s12 > 0.9:
            st.markdown(
                "La survie a 1 an est elevee (> 90%), suggerant un bon pronostic "
                "a court terme pour la population etudiee."
            )
        elif s12 < 0.5:
            st.markdown(
                "La survie a 1 an est inferieure a 50%, indiquant un pronostic "
                "defavorable a court terme."
            )

        # Survival table
        st.markdown("#### Tableau des probabilites de survie")
        st.markdown("Le tableau des probabilités de survie permet de détailler l’évolution de la survie des patients à différents instants du suivi.")
        table, _ = km_survival_table(df, time_col, event_col, [12, 24, 36, 60, 100])
        st.dataframe(table, use_container_width=True, hide_index=True)

        # Full survival table (collapsible)
        with st.expander("Tableau complet des proportions de survivants"):
            kmf_full = KaplanMeierFitter()
            kmf_full.fit(df[time_col], event_observed=df[event_col])
            full_table = kmf_full.survival_function_.copy()
            full_table = full_table.reset_index()
            full_table.columns = ["Temps (mois)", "S(t)"]
            full_table["S(t) %"] = (full_table["S(t)"] * 100).round(2)
            st.dataframe(full_table, use_container_width=True, height=400, hide_index=True)

        # Stratified
        st.markdown("---")
        st.markdown("#### Courbes de survie stratifiées")
        group_label = st.selectbox(
            "Variable de stratification", list(available_groups.keys()), key="km_group"
        )
        group_col = available_groups[group_label]

        col1, col2 = st.columns([2, 1])
        with col1:
            fig = plot_km_stratified(df, time_col, event_col, group_col)
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            st.markdown("**Survie mediane par groupe**")
            medians = km_median_by_group(df, time_col, event_col, group_col)
            st.dataframe(medians, use_container_width=True, hide_index=True)

            # Log-rank
            r = logrank_result(df, time_col, event_col, group_col)
            st.markdown("**Test du Log-Rank**")
            st.info("""
            - H0 : les fonctions de survie sont identiques
            - H1 : au moins un groupe differe
            - Rejet de H0 si **p < 0.05**
            """)
            st.metric("Statistique", f"{r['stat']:.4f}")
            st.metric("p-value", f"{r['p']:.4f}")
            if r["p"] < 0.05:
                st.success(f"Difference significative (p = {r['p']:.4f})")
            else:
                st.warning(f"Difference non significative (p = {r['p']:.4f})")

        # Stratified interpretation
        medians_vals = medians["Mediane (mois)"].astype(float)
        best_group = medians.loc[medians_vals.idxmax(), "Groupe"]
        worst_group = medians.loc[medians_vals.idxmin(), "Groupe"]
        best_val = medians_vals.max()
        worst_val = medians_vals.min()

        interp = (
            f"En stratifiant par **{group_label}**, le groupe **{best_group}** "
            f"presente la survie mediane la plus longue ({best_val:.1f} mois), "
            f"tandis que le groupe **{worst_group}** a la plus courte ({worst_val:.1f} mois). "
        )
        if r["p"] < 0.05:
            interp += (
                f"Le test du Log-Rank confirme que cette difference est **statistiquement "
                f"significative** (p = {r['p']:.4f}), ce qui suggere que la variable "
                f"**{group_label}** a un impact reel sur la survie des patients."
            )
        else:
            interp += (
                f"Cependant, le test du Log-Rank indique que cette difference **n'est pas "
                f"statistiquement significative** (p = {r['p']:.4f}). La variable **{group_label}** ne semble pas avoir un impact sur la survie des patients."
                f" Cela signifie que la différence observée entre les groupes peut être due au hasard."
            )
        st.markdown(interp)

    # ══════════════════════════════════════════════════════════════════════════
    # NELSON-AALEN
    # ══════════════════════════════════════════════════════════════════════════
    elif section == "Nelson-Aalen":
        st.markdown("---")
        st.markdown("### Estimateur de Nelson-Aalen")
        st.markdown("L’estimateur de Nelson-Aalen permet d’estimer le risque cumulé de survenue de l’événement (ici le décès) au cours du temps.")
        st.latex(r"\hat{H}(t) = \sum_{t_i \le t} \frac{d_i}{n_i}")
        st.markdown("Relation avec la survie : $S(t) \\approx e^{-H(t)}$")

        # Global
        st.markdown("#### Risque cumule global")
        st.markdown("Cette courbe permet de visualiser l’accumulation du risque de décès au cours du temps.")
        fig, naf_global = plot_na_global(df, time_col, event_col)
        st.pyplot(fig)
        plt.close(fig)

        h12 = float(naf_global.predict(12))
        h36 = float(naf_global.predict(36))
        h60 = float(naf_global.predict(60))

        col1, col2, col3 = st.columns(3)
        col1.metric("H(12 mois)", f"{h12:.4f}")
        col2.metric("H(36 mois)", f"{h36:.4f}")
        col3.metric("H(60 mois)", f"{h60:.4f}")

        ratio_36_12 = h36 / h12 if h12 > 0 else 0
        st.markdown(
            f"Le risque cumule atteint **{h12:.3f}** a 12 mois et **{h36:.3f}** a 36 mois "
            f"(multiplication par {ratio_36_12:.1f}). "
            f"{"La courbe croît de façon approximativement linéaire, ce qui indique que le taux de risque instantané de deces reste relativement stable dans le temps. Le risque d'événement à chaque instant ne s'accélère ni ne diminue au fil du suivi." if 1.5 < ratio_36_12 < 4.5 else ''}"
            f"{'Le risque accelere fortement apres la premiere annee, ce qui peut indiquer une aggravation progressive de la maladie.' if ratio_36_12 >= 4.5 else ''}"
            f"{'Le risque cumulé progresse lentement, suggerant une population a faible risque dans cette periode.' if ratio_36_12 <= 1.5 else ''}"
        )

        # Estimation interactive
        st.markdown("---")
        st.markdown("#### Estimation de la survie pour un temps donne")
        t_input = st.number_input(
            "Entrez un temps (mois)", min_value=0.0,
            max_value=float(df[time_col].max()), value=24.0, step=1.0,
            key="na_time_input",
        )
        naf_est = NelsonAalenFitter()
        naf_est.fit(df[time_col], event_observed=df[event_col])
        kmf_est = KaplanMeierFitter()
        kmf_est.fit(df[time_col], event_observed=df[event_col])

        import numpy as np
        h_t = float(naf_est.predict(t_input))
        s_t_km = float(kmf_est.predict(t_input))
        s_t_na = np.exp(-h_t)

        col1, col2, col3 = st.columns(3)
        col1.metric(f"H({t_input:.0f}) — Nelson-Aalen", f"{h_t:.4f}")
        col2.metric(f"S({t_input:.0f}) — Kaplan-Meier", f"{s_t_km:.4f} ({s_t_km*100:.2f}%)")
        col3.metric(f"S({t_input:.0f}) ≈ exp(-H(t))", f"{s_t_na:.4f} ({s_t_na*100:.2f}%)")

        diff_pct = abs(s_t_km - s_t_na) * 100
        st.markdown(
            f"A **{t_input:.0f} mois**, la probabilite de survie estimee par Kaplan-Meier est de "
            f"**{s_t_km*100:.2f}%**, tandis que l'approximation via Nelson-Aalen donne "
            f"**{s_t_na*100:.2f}%** (ecart de {diff_pct:.2f} points). "
            f"{'Les deux estimations sont tres proches, ce qui est attendu pour des echantillons de taille suffisante, ce qui confirme la cohérence des estimations.' if diff_pct < 2 else 'L ecart entre les deux methodes est notable, ce qui peut arriver avec des echantillons petits ou des taux de censure eleves.'}"
        )

        # Stratified
        st.markdown("---")
        st.markdown("#### Risque cumule par groupe")
        group_label = st.selectbox(
            "Variable de stratification", list(available_groups.keys()), key="na_group"
        )
        group_col = available_groups[group_label]
        fig = plot_na_stratified(df, time_col, event_col, group_col)
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("Les différences entre les courbes indiquent que certains groupes accumulent un risque de décès plus rapidement que d’autres.")

        # Interpretation
        st.markdown("---")
        with st.expander("Interprétation de la fonction de risque cumulée"):
            st.markdown("""
**Que représente la fonction de risque cumulée H(t) ?**

La fonction de risque cumulée H(t) mesure le risque total de décès accumulé 
jusqu’au temps t. Elle représente la quantité de risque qu’un individu a cumulée 
au cours du temps. Contrairement à la probabilité de survie S(t), cette fonction 
est croissante.

**Comment évolue le risque cumulatif avec le temps ?**

On observe que le risque cumulé de décès augmente avec le temps, ce qui est attendu 
puisque les événements s’accumulent au fil du suivi. Au début, la courbe augmente 
relativement rapidement, ce qui indique un nombre important d'évènements (décès). 
Ensuite, la croissance devient plus régulière, suggérant que le risque instantané de 
décès reste globalement stable dans le temps. Si la courbe de risque cumulée H(t) est 
approximativement linéaire, cela suggère que le risque instantané est constant au 
cours du temps, ce qui correspond à un modèle exponentiel.
""")