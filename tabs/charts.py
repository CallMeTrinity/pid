import streamlit as st
import pandas as pd
import plotly.express as px


def render(df: pd.DataFrame, time_col: str, event_col: str):
    st.subheader("Representations graphiques")

    # ── Time to Event ─────────────────────────────────────────────────────────
    st.markdown(f"#### Distribution de `{time_col}`")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            df, x=time_col, nbins=40,
            title=f"Histogramme — {time_col}",
            color_discrete_sequence=["#1f77b4"],
        )
        med = df[time_col].median()
        fig.add_vline(x=med, line_dash="dash", line_color="red",
                      annotation_text=f"Mediane: {med:.1f}")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.box(
            df, y=time_col, title=f"Boxplot — {time_col}",
            color_discrete_sequence=["#1f77b4"],
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Qualitative variables ─────────────────────────────────────────────────
    st.markdown("---")
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
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(showlegend=False)
            widget.plotly_chart(fig, use_container_width=True)

    # Binary variables as pie charts
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
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            widget.plotly_chart(fig, use_container_width=True)

    # ── Quantitative variables ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Variables quantitatives")

    quant_map = [("Age", "Age (annees)"), ("BMI", "IMC"), ("Comorbidities", "Comorbidites")]
    quant_cols = [(c, l) for c, l in quant_map if c in df.columns]

    if quant_cols:
        cols = st.columns(len(quant_cols))
        for widget, (col_name, label) in zip(cols, quant_cols):
            fig = px.histogram(
                df, x=col_name, nbins=30, title=label,
                labels={col_name: label},
                color_discrete_sequence=["#ff7f0e"],
            )
            mean_v = df[col_name].mean()
            med_v = df[col_name].median()
            fig.add_vline(x=mean_v, line_dash="dash", line_color="blue",
                          annotation_text=f"Moy: {mean_v:.1f}")
            fig.add_vline(x=med_v, line_dash="dot", line_color="red",
                          annotation_text=f"Med: {med_v:.1f}")
            widget.plotly_chart(fig, use_container_width=True)

    # ── Exploration croisee ───────────────────────────────────────────────────
    st.markdown("---")
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
        opacity=.5,
        color_discrete_sequence=px.colors.qualitative.Set1,
    )
    st.plotly_chart(fig, use_container_width=True)
