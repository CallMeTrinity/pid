import streamlit as st
import pandas as pd
import plotly.express as px


def render(df: pd.DataFrame, time_col: str, event_col: str):
    st.markdown("### Vue d'ensemble du jeu de donnees")

    # ── KPI row ────────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Patients", f"{len(df):,}")
    col2.metric("Variables", len(df.columns))
    col3.metric("Evenements", f"{int(df[event_col].sum()):,}")
    censor_rate = (1 - df[event_col].mean()) * 100
    col4.metric("Taux de censure", f"{censor_rate:.1f}%")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Quick distribution overview ────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            df, x=time_col, color=df[event_col].map({0: "Censure", 1: "Evenement"}),
            nbins=40, title="Distribution du temps de suivi",
            color_discrete_map={"Censure": "#6C63FF", "Evenement": "#EF4444"},
            labels={"color": "Statut"},
            opacity=0.8,
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Event vs Censored donut
        event_counts = df[event_col].value_counts().reset_index()
        event_counts.columns = ["Statut", "Effectif"]
        event_counts["Statut"] = event_counts["Statut"].map({0: "Censure", 1: "Evenement"})
        fig = px.pie(
            event_counts, names="Statut", values="Effectif",
            title="Repartition evenement / censure",
            color="Statut",
            color_discrete_map={"Censure": "#6C63FF", "Evenement": "#EF4444"},
            hole=0.4,
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Data preview ───────────────────────────────────────────────────────────
    st.markdown("#### Apercu des donnees")
    st.dataframe(df, use_container_width=True, height=350)

    # ── Variable types ─────────────────────────────────────────────────────────
    with st.expander("Inspection des variables", expanded=False):
        types_df = pd.DataFrame({
            "Variable": df.columns,
            "Type": [str(df[c].dtype) for c in df.columns],
            "Non-null": [int(df[c].notna().sum()) for c in df.columns],
            "Null": [int(df[c].isna().sum()) for c in df.columns],
            "Uniques": [df[c].nunique() for c in df.columns],
        })
        st.dataframe(types_df, use_container_width=True, hide_index=True)

    # ── Duplicates ─────────────────────────────────────────────────────────────
    id_cols = [c for c in df.columns if c not in ["Tranche_Age", "Tranche_BMI"]]
    n_dup = df.duplicated(subset=id_cols).sum()
    if n_dup == 0:
        st.success("Aucun doublon detecte.")
    else:
        st.warning(f"{n_dup} doublon(s) detecte(s).")
