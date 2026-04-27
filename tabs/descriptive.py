import streamlit as st
import pandas as pd
import plotly.express as px

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter"),
)
COLORS = ["#3B82F6", "#06B6D4", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6"]

# Categories to treat as qualitative even when stored as 0/1 ints
QUALI_AS_INT = ["Smoker", "Event_Observed"]

# Pretty labels for axes (map raw → display string)
LABEL_MAPS = {
    "Smoker": {0: "Non", 1: "Oui"},
    "Event_Observed": {0: "Censuré", 1: "Décès"},
    "Sex": {"Male": "Homme", "Female": "Femme", "M": "Homme", "F": "Femme"},
}

# Ordering for categorical axes
CATEGORY_ORDERS = {
    "Physical_Activity": ["Low", "Moderate", "High"],
    "Tranche_Age": ["<50", "50-60", ">60"],
    "Tranche_BMI": ["<18", "18-26", ">26"],
    "Treatment": ["Standard", "Experimental"],
}


def _pretty_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Return a display-friendly version of a column."""
    s = df[col]
    if col in LABEL_MAPS:
        s = s.map(LABEL_MAPS[col]).fillna(s.astype(str))
    else:
        s = s.astype(str)
    return s


def render(df: pd.DataFrame, time_col: str, event_col: str):
    st.markdown("### Statistiques descriptives")

    # ── Variables quantitatives ───────────────────────────────────────────────
    st.markdown("#### Variables quantitatives")
    quant_cols = [c for c in df.select_dtypes(include="number").columns
                  if c not in [event_col] and c not in QUALI_AS_INT]
    if quant_cols:
        desc = df[quant_cols].describe().T
        desc.columns = ["Count", "Moyenne", "Écart-type", "Min",
                        "Q1 (25%)", "Médiane (50%)", "Q3 (75%)", "Max"]
        desc = desc.round(2)
        st.dataframe(desc, use_container_width=True)

        # Interpretation quanti
        bullets = []
        for c in quant_cols:
            s = df[c]
            mean_v, med_v, std_v = s.mean(), s.median(), s.std()
            cv = (std_v / mean_v * 100) if mean_v else 0
            skew = s.skew()
            if abs(skew) < 0.5:
                shape = "symétrique"
            elif skew > 0:
                shape = "asymétrique à droite"
            else:
                shape = "asymétrique à gauche"
            disp = ("faible" if cv < 20 else "modérée" if cv < 40 else "forte")
            bullets.append(
                f"- **{c}** : moyenne {mean_v:.1f}, médiane {med_v:.1f}, "
                f"écart-type {std_v:.1f}, distribution {shape} avec une dispersion {disp} (CV={cv:.0f}%)."
            )
        st.markdown(
            "**Interprétation :** les statistiques ci-dessus résument la tendance "
            "centrale et la dispersion des variables quantitatives.\n" + "\n".join(bullets)
        )
    else:
        st.info("Aucune variable quantitative.")

    # ── Zoom sur time_col ─────────────────────────────────────────────────────
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown(f"#### Zoom sur `{time_col}`")
    col1, col2, col3, col4, col5 = st.columns(5)
    t = df[time_col]
    col1.metric("Moyenne", f"{t.mean():.1f}")
    col2.metric("Médiane", f"{t.median():.1f}")
    col3.metric("Écart-type", f"{t.std():.1f}")
    col4.metric("Min", f"{t.min():.1f}")
    col5.metric("Max", f"{t.max():.1f}")

    # Histogramme du time to event
    fig = px.histogram(
        df, x=time_col, nbins=40,
        title=f"Distribution de {time_col}",
        color_discrete_sequence=[COLORS[0]],
        opacity=0.85,
    )
    fig.add_vline(x=t.median(), line_dash="dash", line_color="#EF4444",
                  annotation_text=f"Médiane: {t.median():.1f}")
    fig.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    skew = t.skew()
    if abs(skew) < 0.5:
        shape_txt = "relativement symétrique"
    elif skew > 0:
        shape_txt = "asymétrique à droite (étalée vers les grandes valeurs)"
    else:
        shape_txt = "asymétrique à gauche (étalée vers les petites valeurs)"

    iqr = t.quantile(0.75) - t.quantile(0.25)
    cv = t.std() / t.mean() * 100 if t.mean() else 0

    st.markdown(
        f"La distribution du temps de suivi est **{shape_txt}** (skewness = {skew:.2f}). "
        f"L'écart interquartile est de **{iqr:.1f} mois** : 50% des patients ont un temps "
        f"de suivi compris entre {t.quantile(0.25):.1f} et {t.quantile(0.75):.1f} mois. "
        f"Le coefficient de variation ({cv:.1f}%) indique une "
        f"{'forte' if cv > 50 else 'modérée' if cv > 30 else 'faible'} dispersion "
        f"des durées de suivi."
    )

    # Histogramme des comorbidités (si dispo)
    if "Comorbidities" in df.columns:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("#### Zoom sur `Comorbidities`")
        counts_c = df["Comorbidities"].value_counts().sort_index().reset_index()
        counts_c.columns = ["Comorbidités", "Effectif"]
        total = counts_c["Effectif"].sum()
        counts_c["%"] = (counts_c["Effectif"] / total * 100).round(1)
        fig = px.bar(
            counts_c, x="Comorbidités", y="Effectif",
            title="Distribution du nombre de comorbidités",
            text="%",
            color_discrete_sequence=[COLORS[1]],
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(**PLOTLY_LAYOUT, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        pct_0 = (df["Comorbidities"] == 0).mean() * 100
        pct_multi = (df["Comorbidities"] >= 2).mean() * 100
        st.markdown(
            f"**{pct_0:.0f}%** des patients n'ont aucune comorbidité, "
            f"**{pct_multi:.0f}%** en ont au moins 2."
        )

    # ── Variables qualitatives ────────────────────────────────────────────────
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### Variables qualitatives")

    cat_cols = [c for c in df.columns
                if df[c].dtype == "object" or df[c].dtype.name == "category"
                or c in QUALI_AS_INT
                or (df[c].nunique() <= 5 and c not in [time_col, "Comorbidities"])]
    cat_cols = [c for c in cat_cols if c not in ["Tranche_Age", "Tranche_BMI"]
                and c != event_col and c in df.columns]
    # Deduplicate while preserving order
    cat_cols = list(dict.fromkeys(cat_cols))

    if not cat_cols:
        st.info("Aucune variable qualitative détectée.")
    else:
        cols_per_row = 3
        for i in range(0, len(cat_cols), cols_per_row):
            row_cols = st.columns(cols_per_row)
            for j, col_name in enumerate(cat_cols[i:i+cols_per_row]):
                with row_cols[j]:
                    display = _pretty_series(df, col_name)
                    counts = display.value_counts()
                    freq = display.value_counts(normalize=True) * 100

                    order = CATEGORY_ORDERS.get(col_name)
                    if order is None and col_name in LABEL_MAPS:
                        # Order by underlying numeric value when possible
                        raw_order = sorted(df[col_name].dropna().unique().tolist(),
                                           key=lambda x: str(x))
                        order = [LABEL_MAPS[col_name].get(v, str(v)) for v in raw_order]

                    index = list(counts.index)
                    if order:
                        index = [o for o in order if o in index] + \
                                [o for o in index if o not in order]

                    freq_df = pd.DataFrame({
                        "Catégorie": index,
                        "Effectif": [int(counts[k]) for k in index],
                        "%": [round(float(freq[k]), 1) for k in index],
                    })
                    st.markdown(f"**{col_name}**")
                    fig = px.bar(
                        freq_df, x="Catégorie", y="Effectif", text="%",
                        color_discrete_sequence=[COLORS[(i + j) % len(COLORS)]],
                        height=280,
                    )
                    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                    fig.update_layout(
                        showlegend=False, **PLOTLY_LAYOUT,
                        margin=dict(l=20, r=20, t=30, b=20),
                        xaxis_title=col_name,
                    )
                    st.plotly_chart(fig, use_container_width=True)

        # Short interpretation aggregée
        quali_lines = []
        for c in cat_cols:
            display = _pretty_series(df, c)
            top_cat = display.value_counts(normalize=True).idxmax()
            top_pct = display.value_counts(normalize=True).max() * 100
            quali_lines.append(f"- **{c}** : la modalité la plus fréquente est "
                               f"*{top_cat}* ({top_pct:.0f}% des patients).")
        st.markdown(
            "**Interprétation :** la répartition des modalités pour chaque variable qualitative "
            "est la suivante.\n" + "\n".join(quali_lines)
        )

    # ── Variables derivees ────────────────────────────────────────────────────
    derived_cols = [d for d in ["Tranche_Age", "Tranche_BMI"] if d in df.columns]
    if derived_cols:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("#### Tranches (variables dérivées)")
        st.caption("Tranches construites à partir des variables quantitatives "
                   "Age et IMC pour faciliter la comparaison entre groupes.")
        cols = st.columns(len(derived_cols))
        for widget, derived in zip(cols, derived_cols):
            with widget:
                order = CATEGORY_ORDERS.get(derived, None)
                counts = df[derived].value_counts()
                if order:
                    counts = counts.reindex(order).dropna()
                else:
                    counts = counts.sort_index()
                freq = (counts / counts.sum() * 100).round(1)
                d_df = pd.DataFrame({
                    "Tranche": counts.index.astype(str),
                    "Effectif": counts.values.astype(int),
                    "%": freq.values,
                })
                fig = px.bar(
                    d_df, x="Tranche", y="Effectif", text="%",
                    title=derived.replace("_", " "),
                    color_discrete_sequence=[COLORS[3]],
                    height=300,
                )
                fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig.update_layout(showlegend=False, **PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)
