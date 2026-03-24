import streamlit as st
import pandas as pd


def render(df: pd.DataFrame, time_col: str, event_col: str):
    st.subheader("Statistiques descriptives")

    # ── Quantitatives ─────────────────────────────────────────────────────────
    st.markdown("#### Variables quantitatives")
    quant_cols = [c for c in df.select_dtypes(include="number").columns
                  if c not in [event_col]]
    if quant_cols:
        desc = df[quant_cols].describe().T
        desc.columns = ["Count", "Moyenne", "Ecart-type", "Min", "Q1 (25%)", "Mediane (50%)", "Q3 (75%)", "Max"]
        desc = desc.round(2)
        st.dataframe(desc, use_container_width=True)
    else:
        st.info("Aucune variable quantitative.")

    # ── Time to Event detail ──────────────────────────────────────────────────
    st.markdown(f"#### Detail : `{time_col}`")
    col1, col2, col3, col4, col5 = st.columns(5)
    t = df[time_col]
    col1.metric("Moyenne", f"{t.mean():.2f}")
    col2.metric("Mediane", f"{t.median():.2f}")
    col3.metric("Ecart-type", f"{t.std():.2f}")
    col4.metric("Min", f"{t.min():.2f}")
    col5.metric("Max", f"{t.max():.2f}")

    # ── Qualitatives ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Variables qualitatives")

    cat_cols = [c for c in df.columns
                if df[c].dtype == "object" or df[c].dtype.name == "category"
                or (df[c].nunique() <= 5 and c not in quant_cols)]
    cat_cols = [c for c in cat_cols if c not in ["Tranche_Age", "Tranche_BMI"]]

    if not cat_cols:
        st.info("Aucune variable qualitative detectee.")
        return

    for col_name in cat_cols:
        with st.expander(f"**{col_name}**", expanded=True):
            counts = df[col_name].value_counts()
            freq = df[col_name].value_counts(normalize=True) * 100
            freq_df = pd.DataFrame({
                "Modalite": counts.index.astype(str),
                "Effectif": counts.values,
                "Frequence (%)": freq.values.round(2),
            })
            st.dataframe(freq_df, use_container_width=True, hide_index=True)

    # ── Derived variables ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Variables derivees")
    for derived in ["Tranche_Age", "Tranche_BMI"]:
        if derived in df.columns:
            with st.expander(f"**{derived}**", expanded=True):
                counts = df[derived].value_counts().sort_index()
                freq = (counts / len(df) * 100).round(2)
                d_df = pd.DataFrame({
                    "Tranche": counts.index.astype(str),
                    "Effectif": counts.values,
                    "Frequence (%)": freq.values,
                })
                st.dataframe(d_df, use_container_width=True, hide_index=True)
