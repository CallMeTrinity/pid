import streamlit as st
import pandas as pd
import numpy as np
import io
from lifelines import KaplanMeierFitter, NelsonAalenFitter, CoxPHFitter
from utils.data_loader import prepare_cox_data, fit_cox_model
import hashlib


def render(df: pd.DataFrame, time_col: str, event_col: str):
    st.markdown("### Export des resultats")
    st.markdown("Telechargez les tableaux et metriques de l'analyse au format CSV.")

    # ── 1. Donnees filtrees ────────────────────────────────────────────────────
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### Donnees filtrees")
    st.markdown(f"Le jeu de donnees apres application des filtres : **{len(df)} patients**.")

    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Telecharger les donnees filtrees (CSV)",
        data=csv_data,
        file_name="donnees_filtrees.csv",
        mime="text/csv",
    )

    # ── 2. Statistiques descriptives ──────────────────────────────────────────
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### Statistiques descriptives")

    quant_cols = [c for c in df.select_dtypes(include="number").columns if c != event_col]
    if quant_cols:
        desc = df[quant_cols].describe().T
        desc.columns = ["Count", "Moyenne", "Ecart-type", "Min", "Q1", "Mediane", "Q3", "Max"]
        desc = desc.round(4)

        st.dataframe(desc, use_container_width=True, height=250)
        st.download_button(
            "Telecharger les statistiques descriptives (CSV)",
            data=desc.to_csv().encode("utf-8"),
            file_name="statistiques_descriptives.csv",
            mime="text/csv",
        )

    # ── 3. Table de survie Kaplan-Meier ───────────────────────────────────────
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### Table de survie (Kaplan-Meier)")

    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col], event_observed=df[event_col])

    km_table = kmf.survival_function_.copy()
    km_table = km_table.reset_index()
    km_table.columns = ["Temps (mois)", "S(t)"]
    km_table["S(t) %"] = (km_table["S(t)"] * 100).round(2)

    ci = kmf.confidence_interval_survival_function_
    ci = ci.reset_index()
    ci.columns = ["Temps (mois)", "IC bas 95%", "IC haut 95%"]
    km_table = km_table.merge(ci, on="Temps (mois)", how="left")
    km_table["IC bas 95%"] = (km_table["IC bas 95%"] * 100).round(2)
    km_table["IC haut 95%"] = (km_table["IC haut 95%"] * 100).round(2)

    st.dataframe(km_table, use_container_width=True, height=250)
    st.download_button(
        "Telecharger la table de survie (CSV)",
        data=km_table.to_csv(index=False).encode("utf-8"),
        file_name="table_survie_kaplan_meier.csv",
        mime="text/csv",
    )

    # ── 4. Metriques KM cles ─────────────────────────────────────────────────
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### Metriques cles de survie")

    time_points = [12, 24, 36, 60, 100]
    metrics_rows = []
    for t in time_points:
        s = float(kmf.predict(t))
        metrics_rows.append({
            "Temps (mois)": t,
            "S(t)": round(s, 4),
            "S(t) %": f"{s*100:.2f}%",
        })
    metrics_rows.append({
        "Temps (mois)": "Mediane",
        "S(t)": "",
        "S(t) %": f"{kmf.median_survival_time_:.2f} mois",
    })

    metrics_df = pd.DataFrame(metrics_rows)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Telecharger les metriques de survie (CSV)",
        data=metrics_df.to_csv(index=False).encode("utf-8"),
        file_name="metriques_survie.csv",
        mime="text/csv",
    )

    # ── 5. Resultats du modele de Cox ─────────────────────────────────────────
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### Resultats du modele de Cox")

    try:
        cox_data = prepare_cox_data(df, time_col, event_col)
        h = hashlib.md5(cox_data.to_json().encode()).hexdigest()
        cph, dropped = fit_cox_model(h, cox_data, time_col, event_col)

        VAR_LABELS = {
            "Age": "Age",
            "Sex_Female": "Sexe (Femme vs Homme)",
            "Smoker": "Fumeur (Oui vs Non)",
            "Treatment_Experimental": "Traitement Exp. vs Standard",
            "Activity_High": "Activite Haute vs Basse",
            "Activity_Moderate": "Activite Moderee vs Basse",
        }

        cox_summary = cph.summary.copy().reset_index()
        cox_summary.columns = [str(c) for c in cox_summary.columns]
        display = cox_summary[["covariate", "coef", "exp(coef)",
                                "exp(coef) lower 95%", "exp(coef) upper 95%", "z", "p"]].copy()
        display.columns = ["Variable", "Coef (beta)", "HR", "IC bas 95%", "IC haut 95%", "z", "p-value"]
        display["Variable"] = display["Variable"].map(VAR_LABELS).fillna(display["Variable"])
        display = display.round(4)

        col1, col2 = st.columns([3, 1])
        with col1:
            st.dataframe(display, use_container_width=True, hide_index=True)
        with col2:
            st.metric("C-index", f"{cph.concordance_index_:.4f}")
            st.metric("Observations", len(cox_data))
            st.metric("Evenements", int(cox_data[event_col].sum()))

        st.download_button(
            "Telecharger les resultats Cox (CSV)",
            data=display.to_csv(index=False).encode("utf-8"),
            file_name="resultats_modele_cox.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.warning(f"Modele de Cox non disponible : {e}")

    # ── 6. Export complet (ZIP) ───────────────────────────────────────────────
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### Export complet")
    st.markdown("Telechargez tous les tableaux en un seul fichier Excel (plusieurs onglets).")

    try:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Donnees filtrees", index=False)
            if quant_cols:
                desc.to_excel(writer, sheet_name="Stats descriptives")
            km_table.to_excel(writer, sheet_name="Table KM", index=False)
            metrics_df.to_excel(writer, sheet_name="Metriques survie", index=False)
            if 'display' in dir():
                display.to_excel(writer, sheet_name="Modele Cox", index=False)

        st.download_button(
            "Telecharger l'export complet (Excel)",
            data=buffer.getvalue(),
            file_name="analyse_survie_complete.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except ImportError:
        st.info("Installez `openpyxl` pour l'export Excel : `pip install openpyxl`")
