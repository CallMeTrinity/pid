import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from lifelines import KaplanMeierFitter, CoxPHFitter
from utils.data_loader import prepare_cox_data, fit_cox_model
import hashlib

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter"),
)
COLORS_A = ["#6C63FF", "#3B82F6"]
COLORS_B = ["#F59E0B", "#EF4444"]


def render(df: pd.DataFrame, time_col: str, event_col: str):
    st.markdown("### Comparateur de sous-groupes")
    st.markdown("""
    Definissez deux sous-populations en choisissant des criteres differents,
    puis comparez leurs courbes de survie et leurs caracteristiques cote a cote.
    """)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Filter builder ─────────────────────────────────────────────────────────
    col_a, col_sep, col_b = st.columns([5, 1, 5])

    with col_a:
        st.markdown("#### Groupe A")
        filters_a = _build_filters(df, time_col, event_col, suffix="a")

    with col_sep:
        st.markdown("<div style='text-align:center; padding-top:8rem; font-size:2rem; opacity:0.3'>vs</div>",
                    unsafe_allow_html=True)

    with col_b:
        st.markdown("#### Groupe B")
        filters_b = _build_filters(df, time_col, event_col, suffix="b")

    # Apply filters
    df_a = _apply_filters(df, filters_a)
    df_b = _apply_filters(df, filters_b)

    # ── KPIs ──────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### Comparaison des populations")

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Groupe A — n", len(df_a))
    col2.metric("A — Evenements", int(df_a[event_col].sum()) if len(df_a) > 0 else 0)
    col3.metric("A — Age moyen", f"{df_a['Age'].mean():.1f}" if "Age" in df_a.columns and len(df_a) > 0 else "N/A")
    col4.metric("Groupe B — n", len(df_b))
    col5.metric("B — Evenements", int(df_b[event_col].sum()) if len(df_b) > 0 else 0)
    col6.metric("B — Age moyen", f"{df_b['Age'].mean():.1f}" if "Age" in df_b.columns and len(df_b) > 0 else "N/A")

    if len(df_a) < 5 or len(df_b) < 5:
        st.warning("Chaque groupe doit contenir au moins 5 patients pour une comparaison fiable.")
        return

    # ── Kaplan-Meier comparison ───────────────────────────────────────────────
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### Courbes de survie (Kaplan-Meier)")

    kmf_a = KaplanMeierFitter()
    kmf_a.fit(df_a[time_col], event_observed=df_a[event_col])
    kmf_b = KaplanMeierFitter()
    kmf_b.fit(df_b[time_col], event_observed=df_b[event_col])

    fig = go.Figure()

    # Group A
    sf_a = kmf_a.survival_function_
    ci_a = kmf_a.confidence_interval_survival_function_
    fig.add_trace(go.Scatter(
        x=sf_a.index, y=sf_a.iloc[:, 0],
        name="Groupe A", line=dict(color=COLORS_A[0], width=3), mode="lines",
    ))
    fig.add_trace(go.Scatter(
        x=list(ci_a.index) + list(ci_a.index[::-1]),
        y=list(ci_a.iloc[:, 1]) + list(ci_a.iloc[:, 0][::-1]),
        fill="toself",
        fillcolor=f"rgba({int(COLORS_A[0][1:3],16)},{int(COLORS_A[0][3:5],16)},{int(COLORS_A[0][5:7],16)},0.12)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
    ))

    # Group B
    sf_b = kmf_b.survival_function_
    ci_b = kmf_b.confidence_interval_survival_function_
    fig.add_trace(go.Scatter(
        x=sf_b.index, y=sf_b.iloc[:, 0],
        name="Groupe B", line=dict(color=COLORS_B[0], width=3), mode="lines",
    ))
    fig.add_trace(go.Scatter(
        x=list(ci_b.index) + list(ci_b.index[::-1]),
        y=list(ci_b.iloc[:, 1]) + list(ci_b.iloc[:, 0][::-1]),
        fill="toself",
        fillcolor=f"rgba({int(COLORS_B[0][1:3],16)},{int(COLORS_B[0][3:5],16)},{int(COLORS_B[0][5:7],16)},0.12)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip",
    ))

    fig.add_hline(y=0.5, line_dash="dot", line_color="rgba(255,255,255,0.3)",
                  annotation_text="S(t) = 0.5")
    fig.update_layout(
        title="Comparaison des courbes de survie",
        xaxis_title="Temps (mois)", yaxis_title="S(t)",
        yaxis=dict(range=[0, 1.05]),
        **PLOTLY_LAYOUT,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Log-rank test ─────────────────────────────────────────────────────────
    from lifelines.statistics import logrank_test
    lr = logrank_test(
        df_a[time_col], df_b[time_col],
        event_observed_A=df_a[event_col], event_observed_B=df_b[event_col],
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Test du Log-Rank — Statistique", f"{lr.test_statistic:.4f}")
    col2.metric("p-value", f"{lr.p_value:.6f}")
    if lr.p_value < 0.05:
        col3.success("Difference significative")
    else:
        col3.warning("Difference non significative")

    # ── Survival metrics comparison ───────────────────────────────────────────
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### Metriques de survie comparees")

    time_points = [12, 24, 36, 60]
    rows = []
    for t in time_points:
        sa = float(kmf_a.predict(t))
        sb = float(kmf_b.predict(t))
        diff = (sa - sb) * 100
        rows.append({
            "Temps (mois)": t,
            "S(t) Groupe A": f"{sa*100:.1f}%",
            "S(t) Groupe B": f"{sb*100:.1f}%",
            "Difference (A-B)": f"{diff:+.1f} pts",
        })
    rows.append({
        "Temps (mois)": "Mediane",
        "S(t) Groupe A": f"{kmf_a.median_survival_time_:.1f} mois",
        "S(t) Groupe B": f"{kmf_b.median_survival_time_:.1f} mois",
        "Difference (A-B)": f"{kmf_a.median_survival_time_ - kmf_b.median_survival_time_:+.1f} mois",
    })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Profile comparison ────────────────────────────────────────────────────
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### Profil demographique compare")

    profile_vars = [
        ("Age", "Age moyen"),
        ("BMI", "IMC moyen"),
        ("Comorbidities", "Comorbidites moy."),
    ]
    cat_vars = [
        ("Sex", "Female", "% Femmes"),
        ("Smoker", 1, "% Fumeurs"),
        ("Treatment", "Experimental", "% Trait. Exp."),
    ]

    profile_rows = []
    for col, label in profile_vars:
        if col in df.columns:
            va = df_a[col].mean() if len(df_a) > 0 else np.nan
            vb = df_b[col].mean() if len(df_b) > 0 else np.nan
            profile_rows.append({
                "Caracteristique": label,
                "Groupe A": f"{va:.1f}",
                "Groupe B": f"{vb:.1f}",
            })
    for col, val, label in cat_vars:
        if col in df.columns:
            pa = (df_a[col] == val).mean() * 100 if len(df_a) > 0 else 0
            pb = (df_b[col] == val).mean() * 100 if len(df_b) > 0 else 0
            profile_rows.append({
                "Caracteristique": label,
                "Groupe A": f"{pa:.1f}%",
                "Groupe B": f"{pb:.1f}%",
            })

    st.dataframe(pd.DataFrame(profile_rows), use_container_width=True, hide_index=True)

    # ── Cox prediction comparison ─────────────────────────────────────────────
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### Prediction Cox — profil moyen de chaque groupe")

    try:
        cox_data = prepare_cox_data(df, time_col, event_col)
        h = hashlib.md5(cox_data.to_json().encode()).hexdigest()
        cph, dropped = fit_cox_model(h, cox_data, time_col, event_col)

        cox_a = prepare_cox_data(df_a, time_col, event_col)
        cox_b = prepare_cox_data(df_b, time_col, event_col)

        model_cols = list(cph.summary.index)
        mean_a = cox_a[model_cols].mean().to_frame().T
        mean_b = cox_b[model_cols].mean().to_frame().T

        sf_cox_a = cph.predict_survival_function(mean_a)
        sf_cox_b = cph.predict_survival_function(mean_b)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sf_cox_a.index, y=sf_cox_a.iloc[:, 0],
            name="Groupe A (profil moyen)",
            line=dict(color=COLORS_A[0], width=3), mode="lines",
        ))
        fig.add_trace(go.Scatter(
            x=sf_cox_b.index, y=sf_cox_b.iloc[:, 0],
            name="Groupe B (profil moyen)",
            line=dict(color=COLORS_B[0], width=3), mode="lines",
        ))
        fig.add_hline(y=0.5, line_dash="dot", line_color="rgba(255,255,255,0.3)")
        fig.update_layout(
            title="Courbe de survie predite (Cox) — profil moyen de chaque groupe",
            xaxis_title="Temps (mois)", yaxis_title="S(t)",
            yaxis=dict(range=[0, 1.05]),
            **PLOTLY_LAYOUT,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15),
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.info(f"Prediction Cox non disponible : {e}")


def _build_filters(df, time_col, event_col, suffix):
    """Build filter widgets for a group and return filter dict."""
    filters = {}

    if "Age" in df.columns:
        age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
        filters["Age"] = st.slider("Age", age_min, age_max, (age_min, age_max), key=f"cmp_age_{suffix}")

    if "Sex" in df.columns:
        opts = sorted(df["Sex"].dropna().unique().tolist())
        filters["Sex"] = st.multiselect("Sexe", opts, default=opts, key=f"cmp_sex_{suffix}")

    if "Smoker" in df.columns:
        opts = sorted(df["Smoker"].dropna().unique().tolist())
        filters["Smoker"] = st.multiselect("Fumeur", opts, default=opts, key=f"cmp_smoker_{suffix}",
                                           format_func=lambda x: "Oui" if x == 1 else "Non")

    if "Treatment" in df.columns:
        opts = sorted(df["Treatment"].dropna().unique().tolist())
        filters["Treatment"] = st.multiselect("Traitement", opts, default=opts, key=f"cmp_treat_{suffix}")

    if "Physical_Activity" in df.columns:
        opts = sorted(df["Physical_Activity"].dropna().unique().tolist())
        filters["Physical_Activity"] = st.multiselect("Activite", opts, default=opts, key=f"cmp_act_{suffix}")

    if "Comorbidities" in df.columns:
        cmin, cmax = int(df["Comorbidities"].min()), int(df["Comorbidities"].max())
        filters["Comorbidities"] = st.slider("Comorbidites", cmin, cmax, (cmin, cmax), key=f"cmp_comorb_{suffix}")

    return filters


def _apply_filters(df, filters):
    """Apply filter dict to dataframe."""
    result = df.copy()
    for col, val in filters.items():
        if col not in result.columns:
            continue
        if isinstance(val, tuple) and len(val) == 2:
            result = result[(result[col] >= val[0]) & (result[col] <= val[1])]
        elif isinstance(val, list):
            result = result[result[col].isin(val)]
    return result
