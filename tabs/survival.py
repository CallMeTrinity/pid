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
    "Activite physique": "Physical_Activity",
    "Fumeur": "Smoker",
    "Tranche d'age": "Tranche_Age",
    "Tranche IMC": "Tranche_BMI",
}


def render(df: pd.DataFrame, time_col: str, event_col: str):
    st.subheader("Probabilites de survie et courbes de survie")

    available_groups = {k: v for k, v in GROUP_OPTIONS.items() if v in df.columns}

    section = st.radio(
        "Section", ["Kaplan-Meier", "Nelson-Aalen", "Tests de comparaison"],
        horizontal=True, key="surv_section",
    )

    # ══════════════════════════════════════════════════════════════════════════
    # KAPLAN-MEIER
    # ══════════════════════════════════════════════════════════════════════════
    if section == "Kaplan-Meier":
        st.markdown("---")
        st.markdown("### Estimateur de Kaplan-Meier")
        st.latex(r"\hat{S}(t) = \prod_{t_i \le t} \left(1 - \frac{d_i}{n_i}\right)")

        # Global curve
        st.markdown("#### Courbe de survie globale")
        fig, kmf_global = plot_km_global(df, time_col, event_col)
        st.pyplot(fig)
        plt.close(fig)

        col1, col2, col3 = st.columns(3)
        col1.metric("Survie mediane", f"{kmf_global.median_survival_time_:.2f} mois")
        col2.metric("Survie a 12 mois", f"{float(kmf_global.predict(12))*100:.2f}%")
        col3.metric("Survie a 36 mois", f"{float(kmf_global.predict(36))*100:.2f}%")

        # Survival table
        st.markdown("#### Tableau des probabilites de survie")
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
        st.markdown("#### Courbes de survie stratifiees")
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
            st.metric("Statistique", f"{r['stat']:.4f}")
            st.metric("p-value", f"{r['p']:.4f}")
            if r["p"] < 0.05:
                st.success(f"Difference significative (p = {r['p']:.4f})")
            else:
                st.warning(f"Difference non significative (p = {r['p']:.4f})")

    # ══════════════════════════════════════════════════════════════════════════
    # NELSON-AALEN
    # ══════════════════════════════════════════════════════════════════════════
    elif section == "Nelson-Aalen":
        st.markdown("---")
        st.markdown("### Estimateur de Nelson-Aalen")
        st.latex(r"\hat{H}(t) = \sum_{t_i \le t} \frac{d_i}{n_i}")
        st.markdown("Relation avec la survie : $S(t) \\approx e^{-H(t)}$")

        # Global
        st.markdown("#### Risque cumule global")
        fig, naf_global = plot_na_global(df, time_col, event_col)
        st.pyplot(fig)
        plt.close(fig)

        col1, col2, col3 = st.columns(3)
        col1.metric("H(12 mois)", f"{float(naf_global.predict(12)):.4f}")
        col2.metric("H(36 mois)", f"{float(naf_global.predict(36)):.4f}")
        col3.metric("H(60 mois)", f"{float(naf_global.predict(60)):.4f}")

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

        # Interpretation
        with st.expander("Interpretation"):
            st.markdown("""
**Que represente la fonction de risque cumulee H(t) ?**

H(t) mesure le risque total accumule jusqu'au temps t. Elle resume la quantite
totale de risque qu'un individu a supporte. Contrairement a S(t), H(t) est une
fonction croissante.

**Comment evolue le risque cumulatif avec le temps ?**

- En debut de suivi, H(t) croit rapidement : le taux de risque instantane
  est eleve (beaucoup d'evenements).
- Le rythme ralentit ensuite : effet de selection — les individus les plus
  fragiles disparaissent tot, laissant une population plus resistante.
- Une courbe H(t) lineaire indiquerait un risque constant (modele exponentiel).
""")

    # ══════════════════════════════════════════════════════════════════════════
    # TESTS DE COMPARAISON
    # ══════════════════════════════════════════════════════════════════════════
    else:
        st.markdown("---")
        st.markdown("### Tests de comparaison des fonctions de survie (Log-Rank)")
        st.markdown("""
Le test du **Log-Rank** (Mantel-Haenszel) compare les fonctions de survie de
deux ou plusieurs groupes :
- H0 : les fonctions de survie sont identiques
- H1 : au moins un groupe differe
- Rejet de H0 si **p < 0.05**
""")

        results_rows = []
        for label, col in available_groups.items():
            if col in df.columns and df[col].nunique() >= 2:
                r = logrank_result(df, time_col, event_col, col)
                results_rows.append({
                    "Variable": label,
                    "Groupes": df[col].nunique(),
                    "Test": r["test"],
                    "Statistique": f"{r['stat']:.4f}",
                    "p-value": f"{r['p']:.4f}",
                    "Significatif": "Oui" if r["p"] < 0.05 else "Non",
                })

        if results_rows:
            st.dataframe(pd.DataFrame(results_rows), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### Comparaison detaillee")
        group_label = st.selectbox(
            "Choisir une variable", list(available_groups.keys()), key="lr_group"
        )
        group_col = available_groups[group_label]

        col1, col2 = st.columns(2)
        with col1:
            fig = plot_km_stratified(df, time_col, event_col, group_col)
            st.pyplot(fig)
            plt.close(fig)
        with col2:
            medians = km_median_by_group(df, time_col, event_col, group_col)
            st.dataframe(medians, use_container_width=True, hide_index=True)

            r = logrank_result(df, time_col, event_col, group_col)
            st.metric("Statistique du test", f"{r['stat']:.4f}")
            st.metric("p-value", f"{r['p']:.6f}")
            if r["p"] < 0.05:
                st.success(f"Les courbes de survie different significativement selon **{group_label}**.")
            else:
                st.warning(f"Pas de difference significative selon **{group_label}**.")
