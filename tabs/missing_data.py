import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_loader import handle_missing


def render(df_raw: pd.DataFrame, df: pd.DataFrame):
    st.subheader("Gestion des donnees manquantes")

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
        st.success("Aucune donnee manquante dans le jeu de donnees.")
    else:
        st.warning(f"{total_missing} valeur(s) manquante(s) detectee(s).")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### Tableau des valeurs manquantes")
        st.dataframe(miss_df, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("#### Visualisation")
        fig = px.bar(
            miss_df[miss_df["Valeurs manquantes"] > 0],
            x="Variable", y="Valeurs manquantes",
            color="% manquant",
            color_continuous_scale="Reds",
            title="Nombre de valeurs manquantes par variable",
        )
        if miss_df["Valeurs manquantes"].sum() > 0:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Aucune valeur manquante a afficher.")

    # Handling options
    st.markdown("---")
    st.markdown("#### Strategies de traitement")

    if total_missing == 0:
        st.info(
            "Le jeu de donnees est complet. Les options ci-dessous "
            "s'appliqueraient si des valeurs manquantes etaient presentes."
        )

    st.markdown("""
| Strategie | Description | Quand l'utiliser |
|-----------|-------------|------------------|
| **Suppression de lignes** | Supprimer les lignes contenant des `NaN` | Peu de valeurs manquantes, donnees MCAR |
| **Suppression de colonnes** | Supprimer les colonnes avec trop de `NaN` | Variable avec > 50% de manquant |
| **Remplacement par la moyenne** | Remplacer `NaN` par la moyenne de la colonne | Variable quantitative, distribution symetrique |
| **Remplacement par la mediane** | Remplacer `NaN` par la mediane | Variable quantitative, distribution asymetrique |
| **Remplacement par le mode** | Remplacer `NaN` par la valeur la plus frequente | Variable qualitative |
""")

    if total_missing > 0:
        strategy = st.selectbox(
            "Strategie a appliquer",
            ["Aucune", "Suppression de lignes", "Suppression de colonnes",
             "Moyenne", "Mediane", "Mode"],
        )
        cols_with_na = [c for c in df_raw.columns if df_raw[c].isna().any()]
        target_cols = st.multiselect("Colonnes cibles", cols_with_na, default=cols_with_na)

        strat_map = {
            "Suppression de lignes": "drop_rows",
            "Suppression de colonnes": "drop_cols",
            "Moyenne": "mean", "Mediane": "median", "Mode": "mode",
        }

        if strategy != "Aucune" and target_cols:
            if st.button("Appliquer le traitement"):
                df_clean = handle_missing(df_raw, strat_map[strategy], target_cols)
                remaining = int(df_clean.isna().sum().sum())
                st.success(
                    f"Traitement applique. "
                    f"Lignes : {len(df_raw)} -> {len(df_clean)}. "
                    f"Valeurs manquantes restantes : {remaining}."
                )
                st.session_state["df_clean"] = df_clean
                st.rerun()
