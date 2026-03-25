import streamlit as st


def render():
    st.markdown("### A propos — Guide lifelines")
    st.markdown("""
    <div class="info-card">
        <h4>Projet Analyse de Survie</h4>
        <p>Master MIAGE M1 — Projet Ingenierie de Donnees (2025-2026)<br>
        Application interactive d'analyse de survie basee sur la librairie
        <b>lifelines</b> et deployee avec <b>Streamlit</b>.</p>
    </div>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # 1. LIBRAIRIES UTILISEES
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### 1. Librairies utilisees")

    libs = [
        ("streamlit", "Framework Web", "Creation de l'interface interactive (widgets, layout, onglets, sidebar)"),
        ("lifelines", "Analyse de survie", "Estimateurs de Kaplan-Meier, Nelson-Aalen, modele de Cox, tests statistiques, modeles parametriques"),
        ("pandas", "Manipulation de donnees", "Chargement CSV, transformations, filtres, aggregations"),
        ("numpy", "Calcul numerique", "Operations vectorisees, fonctions mathematiques"),
        ("plotly", "Visualisation interactive", "Graphiques interactifs (histogrammes, scatter, heatmaps, courbes de survie)"),
        ("matplotlib / seaborn", "Visualisation statique", "Courbes de survie classiques, forest plots"),
        ("scipy", "Statistiques", "Tests statistiques sous-jacents utilises par lifelines"),
    ]

    for lib, role, desc in libs:
        st.markdown(f"**`{lib}`** — *{role}*")
        st.markdown(f"> {desc}")

    # ══════════════════════════════════════════════════════════════════════════
    # 2. FONCTIONS LIFELINES
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### 2. Principales fonctions de la librairie lifelines")

    st.markdown("##### Estimation non parametrique")
    st.markdown("""
| Classe / Fonction | Description | Utilisation dans l'app |
|---|---|---|
| `KaplanMeierFitter` | Estime la fonction de survie S(t) par la methode de Kaplan-Meier | Onglet Survie — courbes globales et stratifiees |
| `KaplanMeierFitter.fit(T, E)` | Ajuste le modele sur les durees T et evenements E | Estimation des probabilites de survie |
| `KaplanMeierFitter.plot_survival_function()` | Trace la courbe de survie avec IC | Visualisation des courbes |
| `KaplanMeierFitter.predict(t)` | Estime S(t) pour un temps donne | Metriques a 12, 24, 36, 60 mois |
| `KaplanMeierFitter.median_survival_time_` | Temps median de survie | KPI affiche dans les metriques |
| `KaplanMeierFitter.survival_function_` | Table complete des probabilites de survie | Export et tableau des survivants |
| `KaplanMeierFitter.confidence_interval_survival_function_` | Intervalles de confiance de S(t) | Bandes de confiance sur les courbes |
""")

    st.markdown("##### Estimation du risque cumule")
    st.markdown("""
| Classe / Fonction | Description | Utilisation dans l'app |
|---|---|---|
| `NelsonAalenFitter` | Estime la fonction de risque cumulee H(t) | Onglet Survie — section Nelson-Aalen |
| `NelsonAalenFitter.fit(T, E)` | Ajuste l'estimateur sur les donnees | Calcul du risque cumule |
| `NelsonAalenFitter.plot_cumulative_hazard()` | Trace H(t) avec IC | Courbe de risque cumule |
| `NelsonAalenFitter.predict(t)` | Estime H(t) pour un temps donne | Estimation interactive (saisie utilisateur) |
""")

    st.markdown("##### Modele de Cox (regression semi-parametrique)")
    st.markdown("""
| Classe / Fonction | Description | Utilisation dans l'app |
|---|---|---|
| `CoxPHFitter` | Modele de Cox a risques proportionnels | Onglet Modele de Cox |
| `CoxPHFitter.fit(df, duration_col, event_col)` | Ajuste le modele multivarie | Estimation des Hazard Ratios |
| `CoxPHFitter.summary` | Tableau des coefficients, HR, IC, p-values | Tableau des resultats et export |
| `CoxPHFitter.predict_survival_function(X)` | Courbe de survie predite pour un profil | Onglets Prediction et Comparateur |
| `CoxPHFitter.plot_partial_effects_on_outcome()` | Courbes ajustees selon une covariable | Courbes de survie ajustees |
| `CoxPHFitter.concordance_index_` | C-index (qualite du modele) | Metrique de performance |
| `CoxPHFitter.compute_residuals(kind=...)` | Residus de martingale, deviance, Schoenfeld | Onglet Avance — residus |
""")

    st.markdown("##### Modeles parametriques")
    st.markdown("""
| Classe / Fonction | Description | Utilisation dans l'app |
|---|---|---|
| `WeibullFitter` | Modele de survie avec distribution de Weibull | Onglet Avance — modeles parametriques |
| `LogNormalFitter` | Modele de survie avec distribution log-normale | Comparaison AIC/BIC |
| `LogLogisticFitter` | Modele de survie avec distribution log-logistique | Comparaison AIC/BIC |
| `*.AIC_` / `*.BIC_` | Criteres de selection de modele | Choix du meilleur modele |
| `*.predict(t)` | Prediction de S(t) parametrique | Superposition sur courbe KM |
""")

    st.markdown("##### Tests statistiques")
    st.markdown("""
| Classe / Fonction | Description | Utilisation dans l'app |
|---|---|---|
| `logrank_test(T1, T2, E1, E2)` | Test du Log-Rank (Mantel-Haenszel) pour 2 groupes | Comparaison de courbes par variable |
| `multivariate_logrank_test(T, group, E)` | Log-Rank pour > 2 groupes | Test global multi-groupes |
| `proportional_hazard_test(cph, df)` | Test de Schoenfeld (proportionnalite) | Verification des hypotheses de Cox |
""")

    # ══════════════════════════════════════════════════════════════════════════
    # 3. ARCHITECTURE DE L'APPLICATION
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### 3. Architecture de l'application")

    st.code("""
pid/
|-- app.py                    # Point d'entree Streamlit
|-- requirements.txt          # Dependances Python
|-- .streamlit/config.toml    # Theme et configuration
|-- data/
|   |-- survival_data_1000.csv  # Jeu de donnees
|-- tabs/                     # Modules des onglets
|   |-- data_viz.py           # Visualisation des donnees
|   |-- missing_data.py       # Gestion des manquantes
|   |-- descriptive.py        # Statistiques descriptives
|   |-- charts.py             # Representations graphiques
|   |-- survival.py           # Kaplan-Meier & Nelson-Aalen
|   |-- prediction.py         # Prediction individuelle
|   |-- cox_model.py          # Modele de Cox
|   |-- comorbidities.py      # Analyse des comorbidites
|   |-- advanced.py           # Analyses avancees
|   |-- comparator.py         # Comparateur de sous-groupes
|   |-- export.py             # Export des resultats
|   |-- about.py              # Guide lifelines (cette page)
|-- utils/
|   |-- data_loader.py        # Chargement, traitement, filtres, Cox
|   |-- plots.py              # Fonctions de visualisation statistique
|-- docs/
    |-- Projet PID 2025-2026.pdf
    |-- Analyse_de_Survie_avec_Python_et_lifelines.ipynb
    """, language="text")

    # ══════════════════════════════════════════════════════════════════════════
    # 4. FLUX DE DONNEES
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### 4. Flux de donnees")

    st.code("""
CSV (upload ou defaut)
  |
  v
load_csv() -----> pd.DataFrame brut
  |
  v
process_data()
  |- Tranche_Age (<50, 50-60, >60)
  |- Tranche_BMI (<18, 18-26, >26)
  |- Suppression doublons (garde Event=1)
  |
  v
handle_missing() [optionnel]
  |- Suppression lignes/colonnes
  |- Imputation (mean, median, mode)
  |
  v
apply_filters() -----> df_filtered
  |- Age, Sex, Smoker, Treatment
  |- Physical_Activity, BMI, Comorbidities
  |
  v
[Onglets d'analyse]
  |- Kaplan-Meier (KaplanMeierFitter)
  |- Nelson-Aalen (NelsonAalenFitter)
  |- Cox (CoxPHFitter)
  |- Parametriques (Weibull, LogNormal, LogLogistic)
  |- Tests (logrank_test, proportional_hazard_test)
    """, language="text")

    # ══════════════════════════════════════════════════════════════════════════
    # 5. ONGLETS
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### 5. Description des onglets")

    onglets = [
        ("Donnees", "Vue d'ensemble du dataset : KPIs (patients, variables, evenements, taux de censure), "
                     "distribution du temps de suivi, apercu des donnees, inspection des types."),
        ("Manquantes", "Detection et traitement des valeurs manquantes : tableau, visualisation, "
                        "strategies de remplacement (suppression, moyenne, mediane, mode)."),
        ("Statistiques", "Statistiques descriptives : variables quantitatives (describe), "
                          "variables qualitatives (effectifs, frequences), variables derivees."),
        ("Graphiques", "Exploration visuelle : histogrammes, boxplots, bar charts, pie charts, "
                        "scatter plots avec coloration par variable categorielle."),
        ("Survie", "Analyse de survie : Kaplan-Meier (globale, stratifiee, tableau des survivants), "
                    "Nelson-Aalen (risque cumule, estimation interactive), tests Log-Rank."),
        ("Prediction", "Prediction individuelle : saisie du profil patient, courbe de survie predite "
                        "par le modele de Cox, comparaison avec profils de reference."),
        ("Modele de Cox", "Regression de Cox : tableau des coefficients et Hazard Ratios, "
                           "forest plot, courbes ajustees, test de Schoenfeld."),
        ("Comorbidites", "Analyse dediee : distribution, impact sur la survie (KM + Log-Rank), "
                          "profil par groupe, interaction avec le traitement."),
        ("Avance", "Modeles parametriques (Weibull, Log-Normal, Log-Logistique) avec AIC/BIC, "
                    "matrice de correlation, residus de Cox, analyse de sensibilite."),
        ("Comparateur", "Comparaison cote-a-cote de deux sous-populations definies par l'utilisateur : "
                         "courbes KM, test Log-Rank, metriques, profil demographique, prediction Cox."),
        ("Export", "Telechargement des resultats : donnees filtrees, statistiques descriptives, "
                    "table de survie, metriques, resultats Cox (CSV et Excel)."),
    ]

    for name, desc in onglets:
        st.markdown(f"**{name}** — {desc}")
