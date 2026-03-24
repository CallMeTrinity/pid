import streamlit as st
import pandas as pd


def render(df: pd.DataFrame, time_col: str, event_col: str):
    st.subheader("Visualisation des donnees")

    col1, col2, col3 = st.columns(3)
    col1.metric("Lignes", len(df))
    col2.metric("Colonnes", len(df.columns))
    col3.metric("Evenements", int(df[event_col].sum()))

    st.markdown("#### Apercu du jeu de donnees")
    st.dataframe(df, use_container_width=True, height=400)

    st.markdown("#### Types des variables")
    types_df = pd.DataFrame({
        "Variable": df.columns,
        "Type": [str(df[c].dtype) for c in df.columns],
        "Non-null": [int(df[c].notna().sum()) for c in df.columns],
        "Null": [int(df[c].isna().sum()) for c in df.columns],
        "Uniques": [df[c].nunique() for c in df.columns],
    })
    st.dataframe(types_df, use_container_width=True, hide_index=True)

    st.markdown("#### Verification des doublons")
    id_cols = [c for c in df.columns if c not in [time_col, event_col, "Tranche_Age", "Tranche_BMI"]]
    n_dup = df.duplicated(subset=id_cols).sum()
    if n_dup == 0:
        st.success("Aucun doublon detecte : chaque patient est present une seule fois.")
    else:
        st.warning(f"{n_dup} doublon(s) detecte(s) (deja supprimes lors du chargement).")
