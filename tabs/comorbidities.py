import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter"),
)
COLORS = ["#10B981", "#3B82F6", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899"]


def render(df: pd.DataFrame, time_col: str, event_col: str):
    if "Comorbidities" not in df.columns:
        st.warning("La variable `Comorbidities` n'est pas presente dans le jeu de donnees.")
        return

    st.markdown("### Analyse des Comorbidites")
    st.markdown("""
    <div class="info-card">
        <h4>Qu'est-ce qu'une comorbidite ?</h4>
        <p>Une comorbidite designe la presence d'une ou plusieurs pathologies associees
        chez un patient. Le nombre de comorbidites est un indicateur majeur de la complexite
        clinique et un facteur pronostique reconnu en analyse de survie.</p>
    </div>
    """, unsafe_allow_html=True)

    comorb = df["Comorbidities"]

    # ══════════════════════════════════════════════════════════════════════════
    # 1. VUE D'ENSEMBLE
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### 1. Vue d'ensemble")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Moyenne", f"{comorb.mean():.2f}")
    col2.metric("Mediane", f"{comorb.median():.0f}")
    col3.metric("Max", f"{int(comorb.max())}")
    pct_multi = (comorb >= 2).mean() * 100
    col4.metric("≥ 2 comorbidites", f"{pct_multi:.1f}%")

    col1, col2 = st.columns(2)

    with col1:
        # Distribution
        counts = comorb.value_counts().sort_index().reset_index()
        counts.columns = ["Comorbidites", "Effectif"]
        fig = px.bar(
            counts, x="Comorbidites", y="Effectif",
            title="Distribution du nombre de comorbidites",
            color="Comorbidites",
            color_continuous_scale="Viridis",
            text="Effectif",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Donut chart by group
        df_temp = df.copy()
        df_temp["Groupe_Comorb"] = pd.cut(
            comorb, bins=[-0.5, 0.5, 1.5, comorb.max() + 0.5],
            labels=["Aucune (0)", "Une (1)", "Multiples (2+)"]
        )
        group_counts = df_temp["Groupe_Comorb"].value_counts().reset_index()
        group_counts.columns = ["Groupe", "Effectif"]
        fig = px.pie(
            group_counts, names="Groupe", values="Effectif",
            title="Repartition par categorie",
            color_discrete_sequence=COLORS,
            hole=0.4,
        )
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # 2. IMPACT SUR LA SURVIE
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### 2. Impact sur la survie")

    col1, col2 = st.columns([2, 1])

    with col1:
        # KM curves by comorbidity count
        fig = go.Figure()
        kmf = KaplanMeierFitter()
        groups = sorted(comorb.unique())

        for i, g in enumerate(groups):
            mask = comorb == g
            kmf.fit(df.loc[mask, time_col], event_observed=df.loc[mask, event_col])
            sf = kmf.survival_function_
            ci = kmf.confidence_interval_survival_function_

            fig.add_trace(go.Scatter(
                x=sf.index, y=sf.iloc[:, 0],
                name=f"{int(g)} comorbidite(s)",
                line=dict(color=COLORS[i % len(COLORS)], width=2.5),
                mode="lines",
            ))
            fig.add_trace(go.Scatter(
                x=list(ci.index) + list(ci.index[::-1]),
                y=list(ci.iloc[:, 1]) + list(ci.iloc[:, 0][::-1]),
                fill="toself",
                fillcolor=COLORS[i % len(COLORS)].replace(")", ",0.1)").replace("rgb", "rgba") if "rgb" in COLORS[i % len(COLORS)] else f"rgba({int(COLORS[i % len(COLORS)][1:3], 16)},{int(COLORS[i % len(COLORS)][3:5], 16)},{int(COLORS[i % len(COLORS)][5:7], 16)},0.1)",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
            ))

        fig.add_hline(y=0.5, line_dash="dot", line_color="rgba(255,255,255,0.3)",
                      annotation_text="S(t) = 0.5")
        fig.update_layout(
            title="Courbes de survie (Kaplan-Meier) par nombre de comorbidites",
            xaxis_title="Temps (mois)",
            yaxis_title="S(t)",
            yaxis=dict(range=[0, 1.05]),
            **PLOTLY_LAYOUT,
            legend=dict(orientation="h", yanchor="bottom", y=-0.25),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Median survival by group
        st.markdown("**Survie mediane par groupe**")
        rows = []
        for g in groups:
            mask = comorb == g
            kmf.fit(df.loc[mask, time_col], event_observed=df.loc[mask, event_col])
            rows.append({
                "Comorbidites": int(g),
                "n": int(mask.sum()),
                "Evenements": int(df.loc[mask, event_col].sum()),
                "Mediane (mois)": f"{kmf.median_survival_time_:.1f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Log-rank test
        st.markdown("**Test du Log-Rank**")
        if len(groups) >= 2:
            r = multivariate_logrank_test(df[time_col], comorb, df[event_col])
            st.metric("Statistique", f"{r.test_statistic:.4f}")
            st.metric("p-value", f"{r.p_value:.6f}")
            if r.p_value < 0.05:
                st.success("Difference significative entre les groupes")
            else:
                st.warning("Difference non significative")

    # ══════════════════════════════════════════════════════════════════════════
    # 3. PROFIL DES PATIENTS PAR COMORBIDITE
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### 3. Profil des patients par niveau de comorbidite")

    df_profil = df.copy()
    df_profil["Groupe_Comorb"] = pd.cut(
        comorb, bins=[-0.5, 0.5, 1.5, comorb.max() + 0.5],
        labels=["0", "1", "2+"]
    )

    col1, col2 = st.columns(2)

    with col1:
        # Age distribution by comorbidity group
        fig = px.box(
            df_profil, x="Groupe_Comorb", y="Age",
            color="Groupe_Comorb",
            title="Age par groupe de comorbidites",
            labels={"Groupe_Comorb": "Comorbidites", "Age": "Age (annees)"},
            color_discrete_sequence=COLORS,
        )
        fig.update_layout(showlegend=False, **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # BMI distribution by comorbidity group
        if "BMI" in df.columns:
            fig = px.box(
                df_profil, x="Groupe_Comorb", y="BMI",
                color="Groupe_Comorb",
                title="IMC par groupe de comorbidites",
                labels={"Groupe_Comorb": "Comorbidites", "BMI": "IMC"},
                color_discrete_sequence=COLORS,
            )
            fig.update_layout(showlegend=False, **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

    # Cross-tabs with categorical variables
    col1, col2, col3 = st.columns(3)

    cat_vars = [
        ("Sex", "Sexe", col1),
        ("Smoker", "Fumeur", col2),
        ("Treatment", "Traitement", col3),
    ]

    for var, label, widget in cat_vars:
        if var not in df.columns:
            continue
        with widget:
            cross = pd.crosstab(
                df_profil["Groupe_Comorb"],
                df[var].astype(str),
                normalize="index",
            ) * 100
            melted = cross.reset_index().melt(id_vars="Groupe_Comorb", var_name=label, value_name="Proportion")
            fig = px.bar(
                melted,
                x="Groupe_Comorb", y="Proportion", color=label,
                title=f"{label} par comorbidites",
                labels={"Groupe_Comorb": "Comorbidites", "Proportion": "Proportion (%)"},
                color_discrete_sequence=COLORS,
                barmode="group",
                text="Proportion",
            )
            fig.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
            fig.update_layout(**PLOTLY_LAYOUT, legend=dict(orientation="h", yanchor="bottom", y=-0.35))
            st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # 4. TEMPS DE SUIVI ET TAUX D'EVENEMENTS
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### 4. Temps de suivi et taux d'evenements")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.violin(
            df_profil, x="Groupe_Comorb", y=time_col,
            color="Groupe_Comorb",
            title="Distribution du temps de suivi par comorbidites",
            labels={"Groupe_Comorb": "Comorbidites", time_col: "Temps (mois)"},
            color_discrete_sequence=COLORS,
            box=True,
        )
        fig.update_layout(showlegend=False, **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Event rate by comorbidity
        event_rates = df_profil.groupby("Groupe_Comorb").agg(
            n=(event_col, "count"),
            events=(event_col, "sum"),
        ).reset_index()
        event_rates["Taux (%)"] = (event_rates["events"] / event_rates["n"] * 100).round(1)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=event_rates["Groupe_Comorb"].astype(str),
            y=event_rates["Taux (%)"],
            marker_color=COLORS[:len(event_rates)],
            text=event_rates["Taux (%)"].apply(lambda x: f"{x:.1f}%"),
            textposition="outside",
        ))
        fig.update_layout(
            title="Taux d'evenements par groupe de comorbidites",
            xaxis_title="Comorbidites",
            yaxis_title="Taux d'evenements (%)",
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # 5. ANALYSE PAIRWISE
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### 5. Comparaisons pairwise (Log-Rank)")
    st.markdown("Test de chaque paire de groupes pour identifier quels niveaux de comorbidites "
                "different significativement.")

    if len(groups) >= 2:
        pairwise_rows = []
        for i_idx in range(len(groups)):
            for j_idx in range(i_idx + 1, len(groups)):
                g1, g2 = groups[i_idx], groups[j_idx]
                m1 = comorb == g1
                m2 = comorb == g2
                r = logrank_test(
                    df.loc[m1, time_col], df.loc[m2, time_col],
                    event_observed_A=df.loc[m1, event_col],
                    event_observed_B=df.loc[m2, event_col],
                )
                pairwise_rows.append({
                    "Groupe A": f"{int(g1)} comorbidite(s)",
                    "Groupe B": f"{int(g2)} comorbidite(s)",
                    "Statistique": f"{r.test_statistic:.4f}",
                    "p-value": f"{r.p_value:.6f}",
                    "Significatif (p<0.05)": "Oui" if r.p_value < 0.05 else "Non",
                })

        st.dataframe(pd.DataFrame(pairwise_rows), use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════════════
    # 6. INTERACTION COMORBIDITES x TRAITEMENT
    # ══════════════════════════════════════════════════════════════════════════
    if "Treatment" in df.columns:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("#### 6. Interaction Comorbidites x Traitement")
        st.markdown("Est-ce que l'effet du traitement varie selon le nombre de comorbidites ?")

        fig = go.Figure()
        kmf = KaplanMeierFitter()

        combo_labels = []
        for g in sorted(df_profil["Groupe_Comorb"].dropna().unique()):
            for treat in sorted(df["Treatment"].unique()):
                mask = (df_profil["Groupe_Comorb"] == g) & (df["Treatment"] == treat)
                if mask.sum() < 5:
                    continue
                label = f"Comorb={g}, {treat}"
                combo_labels.append(label)
                kmf.fit(df.loc[mask, time_col], event_observed=df.loc[mask, event_col])
                sf = kmf.survival_function_
                color_idx = len(combo_labels) - 1
                fig.add_trace(go.Scatter(
                    x=sf.index, y=sf.iloc[:, 0],
                    name=label,
                    line=dict(color=COLORS[color_idx % len(COLORS)], width=2),
                    mode="lines",
                ))

        fig.add_hline(y=0.5, line_dash="dot", line_color="rgba(255,255,255,0.3)")
        fig.update_layout(
            title="Survie par comorbidites et traitement",
            xaxis_title="Temps (mois)",
            yaxis_title="S(t)",
            yaxis=dict(range=[0, 1.05]),
            **PLOTLY_LAYOUT,
            legend=dict(orientation="h", yanchor="bottom", y=-0.35),
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary table
        summary_rows = []
        for g in sorted(df_profil["Groupe_Comorb"].dropna().unique()):
            for treat in sorted(df["Treatment"].unique()):
                mask = (df_profil["Groupe_Comorb"] == g) & (df["Treatment"] == treat)
                if mask.sum() < 2:
                    continue
                kmf.fit(df.loc[mask, time_col], event_observed=df.loc[mask, event_col])
                summary_rows.append({
                    "Comorbidites": str(g),
                    "Traitement": treat,
                    "n": int(mask.sum()),
                    "Evenements": int(df.loc[mask, event_col].sum()),
                    "Mediane survie": f"{kmf.median_survival_time_:.1f}",
                    "Survie 12 mois": f"{float(kmf.predict(12))*100:.1f}%",
                    "Survie 36 mois": f"{float(kmf.predict(36))*100:.1f}%",
                })

        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════════════
    # 7. SYNTHESE
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### Synthese")

    # Compute key findings
    medians_by_group = {}
    for g in groups:
        mask = comorb == g
        kmf.fit(df.loc[mask, time_col], event_observed=df.loc[mask, event_col])
        medians_by_group[int(g)] = kmf.median_survival_time_

    best_group = min(medians_by_group, key=medians_by_group.get)
    worst_group = max(medians_by_group, key=medians_by_group.get)

    event_rate_0 = df.loc[comorb == 0, event_col].mean() * 100 if 0 in groups else None
    event_rate_max = df.loc[comorb == comorb.max(), event_col].mean() * 100

    r_global = multivariate_logrank_test(df[time_col], comorb, df[event_col])

    findings = []
    findings.append(f"- Le nombre moyen de comorbidites est de **{comorb.mean():.2f}** "
                    f"(mediane : {comorb.median():.0f}, max : {int(comorb.max())}).")
    findings.append(f"- **{pct_multi:.1f}%** des patients ont 2 comorbidites ou plus.")

    if r_global.p_value < 0.05:
        findings.append(f"- Le test du Log-Rank global est **significatif** (p = {r_global.p_value:.6f}) : "
                        f"le nombre de comorbidites a un impact sur la survie.")
    else:
        findings.append(f"- Le test du Log-Rank global est **non significatif** (p = {r_global.p_value:.4f}).")

    findings.append(f"- Survie mediane la plus longue : **{worst_group} comorbidite(s)** "
                    f"({medians_by_group[worst_group]:.1f} mois).")
    findings.append(f"- Survie mediane la plus courte : **{best_group} comorbidite(s)** "
                    f"({medians_by_group[best_group]:.1f} mois).")

    if event_rate_0 is not None:
        findings.append(f"- Taux d'evenement : **{event_rate_0:.1f}%** sans comorbidite "
                        f"vs **{event_rate_max:.1f}%** avec {int(comorb.max())} comorbidite(s).")

    for f in findings:
        st.markdown(f)
