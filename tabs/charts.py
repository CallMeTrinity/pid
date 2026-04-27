import streamlit as st
import pandas as pd
import plotly.express as px

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter"),
)

# Coherent palette across the app
COLORS = ["#3B82F6", "#06B6D4", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899"]

LABEL_MAPS = {
    "Smoker": {0: "Non", 1: "Oui"},
    "Event_Observed": {0: "Censuré", 1: "Décès"},
    "Sex": {"Male": "Homme", "Female": "Femme"},
}

CATEGORY_ORDERS = {
    "Physical_Activity": ["Low", "Moderate", "High"],
    "Tranche_Age": ["<50", "50-60", ">60"],
    "Tranche_BMI": ["<18", "18-26", ">26"],
    "Treatment": ["Standard", "Experimental"],
    "Smoker": ["Non", "Oui"],
    "Event_Observed": ["Censuré", "Décès"],
    "Sex": ["Homme", "Femme"],
}


def _pretty(df: pd.DataFrame, col: str) -> pd.Series:
    s = df[col]
    if col in LABEL_MAPS:
        return s.map(LABEL_MAPS[col]).fillna(s.astype(str))
    return s.astype(str)


def _ordered_counts(df: pd.DataFrame, col: str) -> pd.DataFrame:
    disp = _pretty(df, col)
    counts = disp.value_counts()
    total = counts.sum()
    order = CATEGORY_ORDERS.get(col)
    idx = list(counts.index)
    if order:
        idx = [o for o in order if o in idx] + [o for o in idx if o not in order]
    return pd.DataFrame({
        col: idx,
        "Effectif": [int(counts[i]) for i in idx],
        "%": [round(counts[i] / total * 100, 1) for i in idx],
    })


def render(df: pd.DataFrame, time_col: str, event_col: str):
    st.markdown("### Exploration graphique")

    # ── Time to Event ─────────────────────────────────────────────────────────
    st.markdown(f"#### Distribution de `{time_col}`")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            df, x=time_col, nbins=40,
            title=f"Histogramme - {time_col}",
            color_discrete_sequence=[COLORS[0]],
            opacity=0.85,
            histnorm="percent",
            labels={"percent": "%"},
        )
        med = df[time_col].median()
        fig.add_vline(x=med, line_dash="dash", line_color="#EF4444",
                      annotation_text=f"Médiane: {med:.1f}")
        fig.update_layout(**PLOTLY_LAYOUT, yaxis_title="% des patients")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.box(
            df, y=time_col, title=f"Boxplot - {time_col}",
            color_discrete_sequence=[COLORS[1]],
        )
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "La distribution de la variable Time_to_Event est fortement asymétrique à droite : "
        "la majorité des événements surviennent dans les premiers mois (entre 0 et environ 50), "
        "tandis que quelques patients présentent des durées de survie beaucoup plus longues."
    )
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Qualitative variables (bar charts in %) ───────────────────────────────
    st.markdown("#### Variables qualitatives")

    quali_cols = [c for c in ["Sex", "Treatment", "Physical_Activity",
                              "Smoker", "Event_Observed"] if c in df.columns]

    if quali_cols:
        rows = [quali_cols[i:i+3] for i in range(0, len(quali_cols), 3)]
        for row in rows:
            cols = st.columns(len(row))
            for widget, col_name in zip(cols, row):
                data = _ordered_counts(df, col_name)
                fig = px.bar(
                    data, x=col_name, y="%", text="%",
                    title=col_name,
                    color_discrete_sequence=[COLORS[quali_cols.index(col_name) % len(COLORS)]],
                )
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig.update_layout(
                    showlegend=False, **PLOTLY_LAYOUT,
                    yaxis_title="% des patients",
                    xaxis_title="",
                )
                widget.plotly_chart(fig, use_container_width=True)

        # Interpretation qualitative
        lines = []
        for c in quali_cols:
            data = _ordered_counts(df, c)
            top = data.iloc[data["%"].idxmax()]
            lines.append(f"- **{c}** : la modalité dominante est *{top[c]}* "
                         f"({top['%']:.1f}% des patients).")
        st.markdown("**Interprétation :**\n" + "\n".join(lines))

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Quantitative variables with tranches highlighted ──────────────────────
    st.markdown("#### Variables quantitatives")

    quant_map = [("Age", "Âge (années)", [0, 50, 60, 120], ["<50", "50-60", ">60"]),
                 ("BMI", "IMC", [0, 18, 26, 100], ["<18", "18-26", ">26"]),
                 ("Comorbidities", "Comorbidités", None, None)]
    quant_cols = [m for m in quant_map if m[0] in df.columns]

    if quant_cols:
        cols = st.columns(len(quant_cols))
        for widget, (col_name, label, bins, tranche_labels) in zip(cols, quant_cols):
            fig = px.histogram(
                df, x=col_name, nbins=30, title=label,
                labels={col_name: label},
                color_discrete_sequence=[COLORS[2]],
                opacity=0.85,
                histnorm="percent",
            )
            mean_v = df[col_name].mean()
            med_v = df[col_name].median()
            fig.add_vline(x=med_v, line_dash="dot", line_color="#EF4444",
                          annotation_text=f"Med: {med_v:.1f}")
            # Draw tranche borders
            if bins is not None:
                for b, lbl in zip(bins[1:-1], tranche_labels[:-1]):
                    fig.add_vline(x=b, line_dash="dash",
                                  line_color="rgba(255,255,255,0.35)",
                                  annotation_text=f"< {lbl}",
                                  annotation_position="top")
            fig.update_layout(**PLOTLY_LAYOUT, yaxis_title="% des patients")
            widget.plotly_chart(fig, use_container_width=True)

        # Interpretation
        interp_lines = []
        for col_name, label, bins, tranche_labels in quant_cols:
            s = df[col_name]
            interp_lines.append(
                f"- **{label}** : moyenne {s.mean():.1f}, médiane {s.median():.1f} "
                f"(min {s.min():.1f}, max {s.max():.1f})."
            )
        st.markdown("**Interprétation :**\n" + "\n".join(interp_lines))

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Exploration croisee ───────────────────────────────────────────────────
    st.markdown("#### Exploration croisée")

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

    df_plot = df.copy()
    if color_var in LABEL_MAPS:
        df_plot[color_var] = df_plot[color_var].map(LABEL_MAPS[color_var]).fillna(df_plot[color_var].astype(str))
    else:
        df_plot[color_var] = df_plot[color_var].astype(str)

    category_order = None
    if color_var in CATEGORY_ORDERS:
        present = df_plot[color_var].unique().tolist()
        category_order = [c for c in CATEGORY_ORDERS[color_var] if c in present]

    fig = px.scatter(
        df_plot,
        x=x_var, y=time_col, color=color_var,
        title=f"{x_var} vs {time_col}",
        labels={time_col: "Temps de suivi (mois)"},
        opacity=.6,
        color_discrete_sequence=COLORS,
        category_orders={color_var: category_order} if category_order else None,
    )
    fig.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    # Dynamic interpretation for the scatter
    try:
        corr = df[[x_var, time_col]].corr().iloc[0, 1]
    except Exception:
        corr = None

    parts = [f"Ce nuage de points croisé **{x_var}** (axe X) avec la durée de suivi "
             f"**{time_col}** (axe Y), coloré par **{color_var}**."]
    if corr is not None:
        if abs(corr) < 0.1:
            strength = "très faible, quasi inexistante"
        elif abs(corr) < 0.3:
            strength = "faible"
        elif abs(corr) < 0.5:
            strength = "modérée"
        else:
            strength = "forte"
        direction = "positive" if corr > 0 else "négative"
        parts.append(
            f"La corrélation linéaire entre {x_var} et {time_col} est **{strength}** "
            f"({direction}, r = {corr:.2f})."
        )

    # Group median comparison
    grp = df_plot.groupby(color_var)[time_col].median().sort_values(ascending=False)
    if len(grp) >= 2:
        best, worst = grp.index[0], grp.index[-1]
        parts.append(
            f"Le groupe **{best}** présente la médiane de suivi la plus longue "
            f"({grp.iloc[0]:.1f} mois), contre **{worst}** qui a la plus courte "
            f"({grp.iloc[-1]:.1f} mois)."
        )
    st.markdown(" ".join(parts))
