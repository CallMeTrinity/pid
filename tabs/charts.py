import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter"),
)

COLORS = ["#6C63FF", "#3B82F6", "#06B6D4", "#10B981", "#F59E0B", "#EF4444", "#EC4899", "#8B5CF6"]


def render(df: pd.DataFrame, time_col: str, event_col: str):
    st.markdown("### Exploration graphique")

    # ── Time to Event ─────────────────────────────────────────────────────────
    st.markdown(f"#### Distribution de `{time_col}`")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            df, x=time_col, nbins=40,
            title=f"Histogramme — {time_col}",
            color_discrete_sequence=[COLORS[0]],
            opacity=0.85,
        )
        med = df[time_col].median()
        fig.add_vline(x=med, line_dash="dash", line_color="#EF4444",
                      annotation_text=f"Mediane: {med:.1f}")
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.box(
            df, y=time_col, title=f"Boxplot — {time_col}",
            color_discrete_sequence=[COLORS[1]],
        )
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Qualitative variables ─────────────────────────────────────────────────
    st.markdown("#### Variables qualitatives")

    cat_map = {
        "Sex": "Sexe", "Treatment": "Traitement",
        "Physical_Activity": "Activite physique",
    }
    cat_cols = [c for c in cat_map if c in df.columns]

    if cat_cols:
        cols = st.columns(len(cat_cols))
        for widget, col_name in zip(cols, cat_cols):
            counts = df[col_name].value_counts().reset_index()
            counts.columns = [cat_map[col_name], "Effectif"]
            fig = px.bar(
                counts, x=cat_map[col_name], y="Effectif",
                color=cat_map[col_name], text="Effectif",
                title=cat_map[col_name],
                color_discrete_sequence=COLORS,
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(showlegend=False, **PLOTLY_LAYOUT)
            widget.plotly_chart(fig, use_container_width=True)

    # Binary variables as donut charts
    bin_map = {"Smoker": ("Fumeur", {0: "Non", 1: "Oui"}),
               "Event_Observed": ("Evenement", {0: "Censure", 1: "Deces"})}
    bin_cols = [c for c in bin_map if c in df.columns]

    if bin_cols:
        cols = st.columns(len(bin_cols))
        for widget, col_name in zip(cols, bin_cols):
            label, mapping = bin_map[col_name]
            counts = df[col_name].value_counts().reset_index()
            counts.columns = [label, "Effectif"]
            counts[label] = counts[label].map(mapping)
            fig = px.pie(
                counts, names=label, values="Effectif",
                title=label,
                color_discrete_sequence=COLORS,
                hole=0.35,
            )
            fig.update_layout(**PLOTLY_LAYOUT)
            widget.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Quantitative variables ────────────────────────────────────────────────
    st.markdown("#### Variables quantitatives")

    quant_map = [("Age", "Age (annees)"), ("BMI", "IMC"), ("Comorbidities", "Comorbidites")]
    quant_cols = [(c, l) for c, l in quant_map if c in df.columns]

    if quant_cols:
        cols = st.columns(len(quant_cols))
        for widget, (col_name, label) in zip(cols, quant_cols):
            fig = px.histogram(
                df, x=col_name, nbins=30, title=label,
                labels={col_name: label},
                color_discrete_sequence=[COLORS[2]],
                opacity=0.85,
            )
            mean_v = df[col_name].mean()
            med_v = df[col_name].median()
            fig.add_vline(x=mean_v, line_dash="dash", line_color="#3B82F6",
                          annotation_text=f"Moy: {mean_v:.1f}")
            fig.add_vline(x=med_v, line_dash="dot", line_color="#EF4444",
                          annotation_text=f"Med: {med_v:.1f}")
            fig.update_layout(**PLOTLY_LAYOUT)
            widget.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Exploration croisee ───────────────────────────────────────────────────
    st.markdown("#### Exploration croisee")

    num_options = [c for c in df.select_dtypes(include="number").columns if c != event_col]
    cat_options = ["Sex", "Treatment", "Physical_Activity", "Smoker",
                   "Event_Observed", "Tranche_Age", "Tranche_BMI"]
    cat_options = [c for c in cat_options if c in df.columns]

    col1, col2 = st.columns(2)
    with col1:
        x_var = st.selectbox("Variable X", num_options,
                             index=num_options.index("Age") if "Age" in num_options else 0,
                             key="chart_x")
    with col2:
        color_var = st.selectbox("Colorier par", cat_options, key="chart_color")

    fig = px.scatter(
        df.assign(**{color_var: df[color_var].astype(str)}),
        x=x_var, y=time_col, color=color_var,
        title=f"{x_var} vs {time_col}",
        labels={time_col: "Temps de suivi (mois)"},
        opacity=.6,
        color_discrete_sequence=COLORS,
    )
    fig.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)
