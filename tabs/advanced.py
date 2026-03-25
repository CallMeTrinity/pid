import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from lifelines import (
    KaplanMeierFitter, WeibullFitter, LogNormalFitter, LogLogisticFitter,
    CoxPHFitter,
)
from utils.data_loader import prepare_cox_data, fit_cox_model
import hashlib

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter"),
)
COLORS = ["#6C63FF", "#3B82F6", "#06B6D4", "#10B981", "#F59E0B", "#EF4444", "#EC4899", "#8B5CF6"]


def render(df: pd.DataFrame, time_col: str, event_col: str):
    st.markdown("### Analyses avancees")

    section = st.radio(
        "Section", [
            "Modeles parametriques",
            "Correlations",
            "Residus de Cox",
            "Sensibilite",
        ],
        horizontal=True, key="adv_section",
    )

    if section == "Modeles parametriques":
        _render_parametric(df, time_col, event_col)
    elif section == "Correlations":
        _render_correlations(df, time_col, event_col)
    elif section == "Residus de Cox":
        _render_residuals(df, time_col, event_col)
    elif section == "Sensibilite":
        _render_sensitivity(df, time_col, event_col)


# ══════════════════════════════════════════════════════════════════════════════
# 1. MODELES PARAMETRIQUES
# ══════════════════════════════════════════════════════════════════════════════

def _render_parametric(df, time_col, event_col):
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### Modeles parametriques de survie")
    st.markdown("""
    Contrairement a Kaplan-Meier (non parametrique), les modeles parametriques supposent
    une **distribution** pour le temps de survie. Cela permet d'extrapoler au-dela des
    donnees observees et de comparer la qualite d'ajustement via les criteres AIC et BIC.
    """)

    T = df[time_col].values
    E = df[event_col].values

    # Fit models
    models = {}

    kmf = KaplanMeierFitter()
    kmf.fit(T, event_observed=E)

    wf = WeibullFitter()
    wf.fit(T, event_observed=E)
    models["Weibull"] = wf

    lnf = LogNormalFitter()
    lnf.fit(T, event_observed=E)
    models["Log-Normal"] = lnf

    llf = LogLogisticFitter()
    llf.fit(T, event_observed=E)
    models["Log-Logistique"] = llf

    # Comparison table
    st.markdown("##### Comparaison des modeles (AIC / BIC)")

    rows = []
    for name, m in models.items():
        rows.append({
            "Modele": name,
            "AIC": f"{m.AIC_:.2f}",
            "BIC": f"{m.BIC_:.2f}",
            "Log-vraisemblance": f"{m.log_likelihood_:.2f}",
            "Mediane estimee": f"{m.median_survival_time_:.2f} mois",
        })

    comp_df = pd.DataFrame(rows)

    # Find best model
    best_aic = min(models.items(), key=lambda x: x[1].AIC_)
    best_bic = min(models.items(), key=lambda x: x[1].BIC_)

    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    col1.metric("Meilleur modele (AIC)", best_aic[0], f"AIC = {best_aic[1].AIC_:.2f}")
    col2.metric("Meilleur modele (BIC)", best_bic[0], f"BIC = {best_bic[1].BIC_:.2f}")

    st.markdown("""
    <div class="info-card">
        <h4>AIC vs BIC</h4>
        <p><b>AIC</b> (Akaike) favorise le meilleur ajustement aux donnees.
        <b>BIC</b> (Bayesian) penalise davantage la complexite du modele.
        Un AIC/BIC plus faible indique un meilleur modele.
        Si les deux criteres concordent, le choix est clair.</p>
    </div>
    """, unsafe_allow_html=True)

    # Plot: KM vs parametric models
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("##### Courbes de survie : Kaplan-Meier vs modeles parametriques")

    fig = go.Figure()

    # KM step curve
    sf_km = kmf.survival_function_
    fig.add_trace(go.Scatter(
        x=sf_km.index, y=sf_km.iloc[:, 0],
        name="Kaplan-Meier",
        line=dict(color="white", width=2.5, dash="dot"),
        mode="lines",
    ))

    # Parametric curves
    t_range = np.linspace(0.1, T.max(), 300)
    for i, (name, m) in enumerate(models.items()):
        sf = m.predict(t_range)
        fig.add_trace(go.Scatter(
            x=t_range, y=sf,
            name=name,
            line=dict(color=COLORS[i], width=2.5),
            mode="lines",
        ))

    fig.add_hline(y=0.5, line_dash="dot", line_color="rgba(255,255,255,0.3)",
                  annotation_text="S(t) = 0.5")
    fig.update_layout(
        title="Comparaison des modeles de survie",
        xaxis_title="Temps (mois)",
        yaxis_title="S(t)",
        yaxis=dict(range=[0, 1.05]),
        **PLOTLY_LAYOUT,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Model details
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("##### Detail des parametres estimes")

    for name, m in models.items():
        with st.expander(f"Parametres : {name}"):
            st.dataframe(m.summary, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 2. CORRELATIONS
# ══════════════════════════════════════════════════════════════════════════════

def _render_correlations(df, time_col, event_col):
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### Matrice de correlation")
    st.markdown("""
    La matrice de correlation mesure la dependance lineaire entre chaque paire de
    variables numeriques. Un coefficient proche de +1 ou -1 indique une forte
    correlation (positive ou negative), tandis qu'un coefficient proche de 0
    indique l'absence de relation lineaire.
    """)

    # Encode categorical for correlation
    df_encoded = df.copy()
    if "Sex" in df_encoded.columns:
        df_encoded["Sex_num"] = (df_encoded["Sex"] == "Female").astype(int)
    if "Treatment" in df_encoded.columns:
        df_encoded["Treatment_num"] = (df_encoded["Treatment"] == "Experimental").astype(int)
    if "Physical_Activity" in df_encoded.columns:
        act_map = {"Low": 0, "Moderate": 1, "High": 2}
        df_encoded["Activity_num"] = df_encoded["Physical_Activity"].map(act_map)

    num_cols = [c for c in df_encoded.select_dtypes(include="number").columns
                if c not in ["Tranche_Age", "Tranche_BMI"]]

    rename_map = {
        "Sex_num": "Sexe (F=1)",
        "Treatment_num": "Traitement (Exp=1)",
        "Activity_num": "Activite (0-2)",
        "Time_to_Event": "Temps survie",
        "Event_Observed": "Evenement",
        "Comorbidities": "Comorbidites",
        "Age": "Age",
        "BMI": "IMC",
        "Smoker": "Fumeur",
    }

    corr_df = df_encoded[num_cols].rename(columns=rename_map)
    corr = corr_df.corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale=["#EF4444", "#1A1D23", "#6C63FF"],
        zmin=-1, zmax=1,
        title="Matrice de correlation (Pearson)",
        aspect="auto",
    )
    fig.update_layout(**PLOTLY_LAYOUT, height=600)
    st.plotly_chart(fig, use_container_width=True)

    # Key correlations with time_col
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("##### Correlations avec le temps de survie")

    time_label = rename_map.get(time_col, time_col)
    if time_label in corr.columns:
        time_corr = corr[time_label].drop(time_label).sort_values()
        fig = go.Figure()
        colors_bar = [COLORS[5] if v < 0 else COLORS[0] for v in time_corr.values]
        fig.add_trace(go.Bar(
            x=time_corr.values,
            y=time_corr.index,
            orientation="h",
            marker_color=colors_bar,
            text=[f"{v:.3f}" for v in time_corr.values],
            textposition="outside",
        ))
        fig.update_layout(
            title=f"Correlation de chaque variable avec {time_label}",
            xaxis_title="Coefficient de Pearson",
            xaxis=dict(range=[-1, 1]),
            **PLOTLY_LAYOUT,
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Scatter matrix for top variables
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("##### Nuage de points : variables les plus correlees au temps de survie")

    scatter_cols = [c for c in ["Age", "BMI", "Comorbidities", "Smoker"] if c in df.columns]
    if scatter_cols and time_col in df.columns:
        selected = st.multiselect(
            "Variables a afficher", scatter_cols, default=scatter_cols[:3],
            key="corr_scatter_vars"
        )
        if selected:
            fig = make_subplots(rows=1, cols=len(selected),
                                subplot_titles=selected)
            for i, var in enumerate(selected):
                fig.add_trace(go.Scatter(
                    x=df[var], y=df[time_col],
                    mode="markers",
                    marker=dict(
                        color=df[event_col],
                        colorscale=[[0, COLORS[0]], [1, COLORS[5]]],
                        size=4, opacity=0.5,
                    ),
                    name=var,
                    showlegend=False,
                ), row=1, col=i+1)
                fig.update_xaxes(title_text=var, row=1, col=i+1)
                fig.update_yaxes(title_text="Temps (mois)" if i == 0 else "", row=1, col=i+1)

            fig.update_layout(
                title="Relation entre variables et temps de survie (couleur = evenement)",
                **PLOTLY_LAYOUT,
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 3. RESIDUS DE COX
# ══════════════════════════════════════════════════════════════════════════════

def _render_residuals(df, time_col, event_col):
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### Analyse des residus du modele de Cox")
    st.markdown("""
    Les residus permettent d'evaluer la qualite d'ajustement du modele de Cox
    et de detecter les observations mal predites ou influentes.
    """)

    cox_data = prepare_cox_data(df, time_col, event_col)
    h = hashlib.md5(cox_data.to_json().encode()).hexdigest()
    cph, dropped = fit_cox_model(h, cox_data, time_col, event_col)

    if dropped:
        st.info(f"Variables retirees (constantes) : {', '.join(dropped)}")

    # Compute residuals
    # Martingale residuals
    martingale = cph.compute_residuals(cox_data, kind="martingale")
    # Deviance residuals
    deviance = cph.compute_residuals(cox_data, kind="deviance")
    # Schoenfeld residuals
    schoenfeld = cph.compute_residuals(cox_data, kind="schoenfeld")

    VAR_LABELS = {
        "Age": "Age",
        "Sex_Female": "Sexe (Femme)",
        "Smoker": "Fumeur",
        "Treatment_Experimental": "Trait. Experimental",
        "Activity_High": "Activite Haute",
        "Activity_Moderate": "Activite Moderee",
    }

    # ── Martingale ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("##### Residus de martingale")
    st.markdown("""
    Les residus de martingale mesurent l'ecart entre le nombre d'evenements observes
    et le nombre attendu par le modele pour chaque patient.
    - Valeurs proches de 0 : bonne prediction
    - Valeurs positives : evenement survenu plus tot que prevu
    - Valeurs negatives : patient survit plus longtemps que prevu
    """)

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(martingale))),
            y=martingale["martingale"],
            mode="markers",
            marker=dict(color=COLORS[0], size=4, opacity=0.5),
            name="Residus",
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.4)")
        fig.update_layout(
            title="Residus de martingale par observation",
            xaxis_title="Index du patient",
            yaxis_title="Residu de martingale",
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(
            martingale, x="martingale", nbins=50,
            title="Distribution des residus de martingale",
            color_discrete_sequence=[COLORS[0]],
            opacity=0.85,
        )
        fig.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.4)")
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    # Martingale vs covariates (functional form check)
    st.markdown("**Residus de martingale vs covariables** (verification de la forme fonctionnelle)")
    cont_vars = [c for c in ["Age", "Smoker"] if c in cox_data.columns]
    if cont_vars:
        fig = make_subplots(rows=1, cols=len(cont_vars),
                            subplot_titles=[VAR_LABELS.get(v, v) for v in cont_vars])
        for i, var in enumerate(cont_vars):
            fig.add_trace(go.Scatter(
                x=cox_data[var].values,
                y=martingale["martingale"],
                mode="markers",
                marker=dict(color=COLORS[i+1], size=4, opacity=0.4),
                showlegend=False,
            ), row=1, col=i+1)
            fig.update_xaxes(title_text=VAR_LABELS.get(var, var), row=1, col=i+1)
            fig.update_yaxes(title_text="Residu" if i == 0 else "", row=1, col=i+1)

        fig.update_layout(
            title="Forme fonctionnelle des covariables continues",
            **PLOTLY_LAYOUT, height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div class="info-card">
            <h4>Lecture</h4>
            <p>Si la relation est lineaire (nuage horizontal autour de 0), la forme
            fonctionnelle est correcte. Une courbe suggere qu'une transformation
            (log, polynome) pourrait ameliorer le modele.</p>
        </div>
        """, unsafe_allow_html=True)

    # ── Deviance ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("##### Residus de deviance")
    st.markdown("""
    Les residus de deviance sont une transformation des residus de martingale
    qui produit une distribution plus symetrique. Les valeurs extremes
    (|residu| > 2) signalent des observations mal ajustees.
    """)

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        outlier_mask = deviance["deviance"].abs() > 2
        fig.add_trace(go.Scatter(
            x=list(range(len(deviance))),
            y=deviance["deviance"],
            mode="markers",
            marker=dict(
                color=[COLORS[5] if o else COLORS[0] for o in outlier_mask],
                size=4, opacity=0.6,
            ),
            name="Residus",
        ))
        fig.add_hline(y=2, line_dash="dash", line_color=COLORS[5],
                      annotation_text="Seuil +2")
        fig.add_hline(y=-2, line_dash="dash", line_color=COLORS[5],
                      annotation_text="Seuil -2")
        fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)")
        fig.update_layout(
            title="Residus de deviance par observation",
            xaxis_title="Index du patient",
            yaxis_title="Residu de deviance",
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        n_outliers = int(outlier_mask.sum())
        pct_outliers = n_outliers / len(deviance) * 100

        st.metric("Observations atypiques (|d| > 2)", f"{n_outliers}")
        st.metric("Proportion", f"{pct_outliers:.1f}%")

        if pct_outliers < 5:
            st.success(
                f"Seulement {pct_outliers:.1f}% d'observations atypiques. "
                "Le modele s'ajuste bien aux donnees."
            )
        else:
            st.warning(
                f"{pct_outliers:.1f}% d'observations atypiques. "
                "Le modele pourrait etre ameliore."
            )

        fig = px.histogram(
            deviance, x="deviance", nbins=50,
            title="Distribution des residus de deviance",
            color_discrete_sequence=[COLORS[1]],
            opacity=0.85,
        )
        fig.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.4)")
        fig.update_layout(**PLOTLY_LAYOUT, height=300)
        st.plotly_chart(fig, use_container_width=True)

    # ── Schoenfeld ────────────────────────────────────────────────────────
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("##### Residus de Schoenfeld")
    st.markdown("""
    Les residus de Schoenfeld testent l'hypothese de **risques proportionnels**.
    Si les residus montrent une tendance en fonction du temps, l'hypothese
    de proportionnalite est violee pour cette covariable.
    """)

    schoenfeld_cols = [c for c in schoenfeld.columns if c != time_col]
    selected_var = st.selectbox(
        "Covariable a examiner",
        schoenfeld_cols,
        format_func=lambda x: VAR_LABELS.get(x, x),
        key="schoenfeld_var",
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=schoenfeld.index,
        y=schoenfeld[selected_var],
        mode="markers",
        marker=dict(color=COLORS[2], size=5, opacity=0.5),
        name="Residus de Schoenfeld",
    ))

    # Add LOWESS-like trend via rolling mean
    sch_sorted = schoenfeld[[selected_var]].copy()
    sch_sorted = sch_sorted.sort_index()
    window = max(len(sch_sorted) // 20, 10)
    trend = sch_sorted[selected_var].rolling(window=window, center=True, min_periods=1).mean()
    fig.add_trace(go.Scatter(
        x=sch_sorted.index,
        y=trend,
        mode="lines",
        line=dict(color=COLORS[5], width=3),
        name="Tendance (moyenne mobile)",
    ))

    fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)")
    fig.update_layout(
        title=f"Residus de Schoenfeld : {VAR_LABELS.get(selected_var, selected_var)}",
        xaxis_title="Temps (mois)",
        yaxis_title="Residu de Schoenfeld",
        **PLOTLY_LAYOUT,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="info-card">
        <h4>Interpretation</h4>
        <p>Si la ligne de tendance est horizontale (plate autour de 0), l'hypothese
        de proportionnalite est respectee. Une pente ou une courbe indique que
        l'effet de la variable change au cours du temps.</p>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 4. ANALYSE DE SENSIBILITE
# ══════════════════════════════════════════════════════════════════════════════

def _render_sensitivity(df, time_col, event_col):
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### Analyse de sensibilite")
    st.markdown("""
    L'analyse de sensibilite evalue la **robustesse** des resultats en excluant
    certains sous-groupes. Si les conclusions changent fortement quand on retire
    un groupe, cela signifie que les resultats dependent de ce sous-groupe.
    """)

    # Define exclusion scenarios
    scenarios = [{"label": "Population complete", "filter": lambda d: d}]

    if "Smoker" in df.columns:
        scenarios.append({"label": "Sans fumeurs", "filter": lambda d: d[d["Smoker"] == 0]})
    if "Age" in df.columns:
        scenarios.append({"label": "Sans patients > 70 ans", "filter": lambda d: d[d["Age"] <= 70]})
    if "Comorbidities" in df.columns:
        scenarios.append({"label": "Sans comorbidites multiples (>=2)",
                          "filter": lambda d: d[d["Comorbidities"] < 2]})
    if "Treatment" in df.columns:
        scenarios.append({"label": "Traitement standard uniquement",
                          "filter": lambda d: d[d["Treatment"] == "Standard"]})
        scenarios.append({"label": "Traitement experimental uniquement",
                          "filter": lambda d: d[d["Treatment"] == "Experimental"]})

    # KM curves for each scenario
    st.markdown("##### Courbes de Kaplan-Meier par scenario")

    fig = go.Figure()
    summary_rows = []

    for i, sc in enumerate(scenarios):
        sub = sc["filter"](df)
        if len(sub) < 10:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(sub[time_col], event_observed=sub[event_col])
        sf = kmf.survival_function_

        fig.add_trace(go.Scatter(
            x=sf.index, y=sf.iloc[:, 0],
            name=sc["label"],
            line=dict(color=COLORS[i % len(COLORS)], width=2.5,
                      dash="solid" if i == 0 else "dash"),
            mode="lines",
        ))

        s12 = float(kmf.predict(12))
        s36 = float(kmf.predict(36))
        s60 = float(kmf.predict(60))

        summary_rows.append({
            "Scenario": sc["label"],
            "n": len(sub),
            "Evenements": int(sub[event_col].sum()),
            "Mediane survie": f"{kmf.median_survival_time_:.1f}",
            "S(12 mois)": f"{s12*100:.1f}%",
            "S(36 mois)": f"{s36*100:.1f}%",
            "S(60 mois)": f"{s60*100:.1f}%",
        })

    fig.add_hline(y=0.5, line_dash="dot", line_color="rgba(255,255,255,0.3)")
    fig.update_layout(
        title="Impact de l'exclusion de sous-groupes sur la survie",
        xaxis_title="Temps (mois)",
        yaxis_title="S(t)",
        yaxis=dict(range=[0, 1.05]),
        **PLOTLY_LAYOUT,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Tableau comparatif des scenarios")
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # Cox model sensitivity
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("##### Stabilite des Hazard Ratios du modele de Cox")
    st.markdown("""
    On compare les Hazard Ratios du modele de Cox entre la population complete
    et chaque scenario d'exclusion. Des HR stables indiquent des resultats robustes.
    """)

    hr_data = []
    for sc in scenarios:
        sub = sc["filter"](df)
        if len(sub) < 30:
            continue
        try:
            cox_sub = prepare_cox_data(sub, time_col, event_col)
            feature_cols = [c for c in cox_sub.columns if c not in [time_col, event_col]]
            varying = [c for c in feature_cols if cox_sub[c].nunique() > 1]
            cox_sub = cox_sub[varying + [time_col, event_col]]

            cph = CoxPHFitter(penalizer=0.01)
            cph.fit(cox_sub, duration_col=time_col, event_col=event_col)

            for var_name in cph.summary.index:
                hr_data.append({
                    "Scenario": sc["label"],
                    "Variable": var_name,
                    "HR": cph.summary.loc[var_name, "exp(coef)"],
                    "IC_low": cph.summary.loc[var_name, "exp(coef) lower 95%"],
                    "IC_high": cph.summary.loc[var_name, "exp(coef) upper 95%"],
                    "p": cph.summary.loc[var_name, "p"],
                })
        except Exception:
            continue

    if hr_data:
        hr_df = pd.DataFrame(hr_data)

        VAR_LABELS = {
            "Age": "Age",
            "Sex_Female": "Sexe (Femme)",
            "Smoker": "Fumeur",
            "Treatment_Experimental": "Trait. Experimental",
            "Activity_High": "Activite Haute",
            "Activity_Moderate": "Activite Moderee",
        }

        variables = hr_df["Variable"].unique()
        selected_var = st.selectbox(
            "Variable a comparer entre scenarios",
            variables,
            format_func=lambda x: VAR_LABELS.get(x, x),
            key="sensitivity_var",
        )

        var_df = hr_df[hr_df["Variable"] == selected_var].copy()

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=var_df["Scenario"],
            y=var_df["HR"],
            marker_color=[COLORS[i % len(COLORS)] for i in range(len(var_df))],
            error_y=dict(
                type="data",
                symmetric=False,
                array=(var_df["IC_high"] - var_df["HR"]).tolist(),
                arrayminus=(var_df["HR"] - var_df["IC_low"]).tolist(),
                color="white",
                thickness=1.5,
            ),
            text=[f"{hr:.3f}" for hr in var_df["HR"]],
            textposition="outside",
        ))
        fig.add_hline(y=1, line_dash="dash", line_color="rgba(255,255,255,0.4)",
                      annotation_text="HR = 1 (neutre)")
        fig.update_layout(
            title=f"Hazard Ratio de '{VAR_LABELS.get(selected_var, selected_var)}' par scenario",
            xaxis_title="Scenario",
            yaxis_title="Hazard Ratio",
            **PLOTLY_LAYOUT,
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Summary
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("##### Synthese")
    st.markdown("""
    <div class="info-card">
        <h4>Comment interpreter</h4>
        <p>Si les courbes de survie et les Hazard Ratios restent similaires entre les
        scenarios, les resultats sont <b>robustes</b>. Si un scenario change fortement
        les conclusions, ce sous-groupe est <b>influent</b> et merite une attention
        particuliere dans l'interpretation.</p>
    </div>
    """, unsafe_allow_html=True)
