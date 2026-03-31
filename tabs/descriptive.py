import streamlit as st
import pandas as pd
import plotly.express as px

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter"),
)
COLORS = ["#6C63FF", "#3B82F6", "#06B6D4", "#10B981", "#F59E0B", "#EF4444"]


def render(df: pd.DataFrame, time_col: str, event_col: str):
    st.markdown("### Statistiques descriptives")

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
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown(f"#### Zoom sur `{time_col}`")
    col1, col2, col3, col4, col5 = st.columns(5)
    t = df[time_col]
    col1.metric("Moyenne", f"{t.mean():.1f}")
    col2.metric("Mediane", f"{t.median():.1f}")
    col3.metric("Ecart-type", f"{t.std():.1f}")
    col4.metric("Min", f"{t.min():.1f}")
    col5.metric("Max", f"{t.max():.1f}")

    # ── Dynamic interpretation of time variable ─────────────────────────────
    skew = t.skew()
    if abs(skew) < 0.5:
        shape_txt = "relativement symetrique"
    elif skew > 0:
        shape_txt = "asymetrique a droite (etalee vers les grandes valeurs)"
    else:
        shape_txt = "asymetrique a gauche (etalee vers les petites valeurs)"

    iqr = t.quantile(0.75) - t.quantile(0.25)
    cv = t.std() / t.mean() * 100

    st.markdown(
        f"La distribution du temps de suivi est **{shape_txt}** (skewness = {skew:.2f}). "
        f"L'ecart interquartile est de **{iqr:.1f} mois** (50% des patients ont un temps "
        f"de suivi compris entre {t.quantile(0.25):.1f} et {t.quantile(0.75):.1f} mois). "
        f"Le coefficient de variation ({cv:.1f}%) indique une "
        f"{'forte' if cv > 50 else 'moderee' if cv > 30 else 'faible'} dispersion "
        f"des durees de suivi."
    )

    # ── Qualitatives ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### Variables qualitatives")

    cat_cols = [c for c in df.columns
                if df[c].dtype == "object" or df[c].dtype.name == "category"
                or (df[c].nunique() <= 5 and c not in quant_cols)]
    cat_cols = [c for c in cat_cols if c not in ["Tranche_Age", "Tranche_BMI"]]

    if not cat_cols:
        st.info("Aucune variable qualitative detectee.")
        return

    # Show as a grid of mini bar charts + tables
    cols_per_row = 3
    for i in range(0, len(cat_cols), cols_per_row):
        row_cols = st.columns(cols_per_row)
        for j, col_name in enumerate(cat_cols[i:i+cols_per_row]):
            with row_cols[j]:
                counts = df[col_name].value_counts()
                freq = df[col_name].value_counts(normalize=True) * 100
                freq_df = pd.DataFrame({
                    "": counts.index.astype(str),
                    "Effectif": counts.values,
                    "% ": freq.values.round(1),
                })
                st.markdown(f"**{col_name}**")
                fig = px.bar(
                    freq_df, x="", y="Effectif", text="% ",
                    color_discrete_sequence=[COLORS[i % len(COLORS)]],
                    height=250,
                )
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig.update_layout(showlegend=False, **PLOTLY_LAYOUT,
                                  margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig, use_container_width=True)

    # ── Derived variables ─────────────────────────────────────────────────────
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### Variables derivees")

    derived_cols = [d for d in ["Tranche_Age", "Tranche_BMI"] if d in df.columns]
    if derived_cols:
        cols = st.columns(len(derived_cols))
        for widget, derived in zip(cols, derived_cols):
            with widget:
                counts = df[derived].value_counts().sort_index()
                freq = (counts / len(df) * 100).round(1)
                d_df = pd.DataFrame({
                    "Tranche": counts.index.astype(str),
                    "Effectif": counts.values,
                    "% ": freq.values,
                })
                fig = px.bar(
                    d_df, x="Tranche", y="Effectif", text="% ",
                    title=derived.replace("_", " "),
                    color_discrete_sequence=[COLORS[3]],
                    height=300,
                )
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig.update_layout(showlegend=False, **PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)
