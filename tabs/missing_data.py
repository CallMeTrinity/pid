import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_loader import handle_missing

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter"),
)


def render(df_raw: pd.DataFrame, df: pd.DataFrame):
    st.markdown("### Gestion des données manquantes")

    # Current state
    missing = df_raw.isna().sum()
    missing_pct = (missing / len(df_raw) * 100).round(2)
    miss_df = pd.DataFrame({
        "Variable": missing.index,
        "Valeurs manquantes": missing.values,
        "% manquant": missing_pct.values,
    }).sort_values("Valeurs manquantes", ascending=False)

    total_missing = int(missing.sum())

    if total_missing == 0:
        st.success("Aucune donnée manquante dans le jeu de données.")
    else:
        st.error(f"{total_missing} valeur(s) manquante(s) détectée(s).")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### Tableau des valeurs manquantes")
        st.dataframe(miss_df, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("#### Visualisation")
        if miss_df["Valeurs manquantes"].sum() > 0:
            fig = px.bar(
                miss_df[miss_df["Valeurs manquantes"] > 0],
                x="Variable", y="Valeurs manquantes",
                color="% manquant",
                color_continuous_scale=["#6C63FF", "#EF4444"],
                title="Valeurs manquantes par variable",
            )
            fig.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("""
            <div class="info-card">
                <h4>Dataset complet</h4>
                <p>Aucune valeur manquante à afficher. Le jeu de données est valide pour l'analyse.</p>
            </div>
            """, unsafe_allow_html=True)

    # Handling options
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### Stratégies de traitement")

    if total_missing == 0:
        st.info(
            "Le jeu de données est complet. Les options ci-dessous "
            "s'appliqueraient si des valeurs manquantes étaient présentes."
        )

    st.markdown("""
| Stratégie | Description | Quand l'utiliser |
|-----------|-------------|------------------|
| **Suppression de lignes** | Supprimer les lignes contenant des `NaN` | Peu de valeurs manquantes, données MCAR |
| **Suppression de colonnes** | Supprimer les colonnes avec trop de `NaN` | Variable avec > 50% de manquant |
| **Remplacement par la moyenne** | Remplacer `NaN` par la moyenne de la colonne | Variable quantitative, distribution symétrique |
| **Remplacement par la médiane** | Remplacer `NaN` par la médiane | Variable quantitative, distribution asymétrique |
| **Remplacement par le mode** | Remplacer `NaN` par la valeur la plus fréquente | Variable qualitative |
""")

    if total_missing > 0:
        strategy = st.selectbox(
            "Stratégie à appliquer",
            ["Aucune", "Suppression de lignes", "Suppression de colonnes",
             "Moyenne", "Médiane", "Mode"],
        )
        cols_with_na = [c for c in df_raw.columns if df_raw[c].isna().any()]
        target_cols = st.multiselect("Colonnes cibles", cols_with_na, default=cols_with_na)

        strat_map = {
            "Suppression de lignes": "drop_rows",
            "Suppression de colonnes": "drop_cols",
            "Moyenne": "mean", "Médiane": "median", "Mode": "mode",
        }

        if strategy != "Aucune" and target_cols:
            if st.button("Appliquer le traitement", type="primary"):
                df_clean = handle_missing(df_raw, strat_map[strategy], target_cols)
                remaining = int(df_clean.isna().sum().sum())
                st.success(
                    f"Traitement appliqué. "
                    f"Lignes : {len(df_raw)} → {len(df_clean)}. "
                    f"Valeurs manquantes restantes : {remaining}."
                )
                st.session_state["df_clean"] = df_clean
                st.rerun()
