import streamlit as st


def render():
    st.markdown("### À propos - Guide lifelines")
    st.markdown("""
    <div class="info-card">
        <h4>Projet Analyse de Survie</h4>
        <p>Master MIAGE M1 - Projet Ingénierie de Données (2025-2026)<br>
        Application interactive d'analyse de survie basée sur la librairie
        <b>lifelines</b> et déployée avec <b>Streamlit</b>.</p>
    </div>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # 1. LIBRAIRIES UTILISEES
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### 1. Librairies utilisées")

    libs = [
        ("streamlit", "Framework Web", "Création de l'interface interactive (widgets, layout, onglets, sidebar)"),
        ("lifelines", "Analyse de survie", "Estimateurs de Kaplan-Meier, Nelson-Aalen, modèle de Cox, tests statistiques, modèles paramétriques"),
        ("pandas", "Manipulation de données", "Chargement CSV, transformations, filtres, agrégations"),
        ("numpy", "Calcul numérique", "Opérations vectorisées, fonctions mathématiques"),
        ("plotly", "Visualisation interactive", "Graphiques interactifs (histogrammes, scatter, heatmaps, courbes de survie)"),
        ("matplotlib / seaborn", "Visualisation statique", "Courbes de survie classiques, forest plots"),
        ("scipy", "Statistiques", "Tests statistiques sous-jacents utilisés par lifelines"),
    ]

    for lib, role, desc in libs:
        st.markdown(f"**`{lib}`** - *{role}*")
        st.markdown(f"> {desc}")

    # ══════════════════════════════════════════════════════════════════════════
    # 2. FONCTIONS LIFELINES
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### 2. Principales fonctions de la librairie lifelines")

    st.markdown("##### Estimation non paramétrique")
    st.markdown("""
| Classe / Fonction | Description | Utilisation dans l'app |
|---|---|---|
| `KaplanMeierFitter` | Estime la fonction de survie S(t) par la méthode de Kaplan-Meier | Onglet Survie - courbes globales et stratifiées |
| `KaplanMeierFitter.fit(T, E)` | Ajuste le modèle sur les durées T et événements E | Estimation des probabilités de survie |
| `KaplanMeierFitter.plot_survival_function()` | Trace la courbe de survie avec IC | Visualisation des courbes |
| `KaplanMeierFitter.predict(t)` | Estime S(t) pour un temps donné | Métriques à 12, 24, 36, 60 mois |
| `KaplanMeierFitter.median_survival_time_` | Temps médian de survie | KPI affiché dans les métriques |
| `KaplanMeierFitter.survival_function_` | Table complète des probabilités de survie | Export et tableau des survivants |
| `KaplanMeierFitter.confidence_interval_survival_function_` | Intervalles de confiance de S(t) | Bandes de confiance sur les courbes |
""")

    st.markdown("##### Estimation du risque cumulé")
    st.markdown("""
| Classe / Fonction | Description | Utilisation dans l'app |
|---|---|---|
| `NelsonAalenFitter` | Estime la fonction de risque cumulée H(t) | Onglet Survie - section Nelson-Aalen |
| `NelsonAalenFitter.fit(T, E)` | Ajuste l'estimateur sur les données | Calcul du risque cumulé |
| `NelsonAalenFitter.plot_cumulative_hazard()` | Trace H(t) avec IC | Courbe de risque cumulé |
| `NelsonAalenFitter.predict(t)` | Estime H(t) pour un temps donné | Estimation interactive (saisie utilisateur) |
""")

    st.markdown("##### Modèle de Cox (régression semi-paramétrique)")
    st.markdown("""
| Classe / Fonction | Description | Utilisation dans l'app |
|---|---|---|
| `CoxPHFitter` | Modèle de Cox à risques proportionnels | Onglet Modèle de Cox |
| `CoxPHFitter.fit(df, duration_col, event_col)` | Ajuste le modèle multivarié | Estimation des Hazard Ratios |
| `CoxPHFitter.summary` | Tableau des coefficients, HR, IC, p-values | Tableau des résultats et export |
| `CoxPHFitter.predict_survival_function(X)` | Courbe de survie prédite pour un profil | Onglets Prédiction et Comparateur |
| `CoxPHFitter.plot_partial_effects_on_outcome()` | Courbes ajustées selon une covariable | Courbes de survie ajustées |
| `CoxPHFitter.concordance_index_` | C-index (qualité du modèle) | Métrique de performance |
| `CoxPHFitter.compute_residuals(kind=...)` | Résidus de martingale, déviance, Schoenfeld | Onglet Avancé - résidus |
""")

    st.markdown("##### Modèles paramétriques")
    st.markdown("""
| Classe / Fonction | Description | Utilisation dans l'app |
|---|---|---|
| `WeibullFitter` | Modèle de survie avec distribution de Weibull | Onglet Avancé - modèles paramétriques |
| `LogNormalFitter` | Modèle de survie avec distribution log-normale | Comparaison AIC/BIC |
| `LogLogisticFitter` | Modèle de survie avec distribution log-logistique | Comparaison AIC/BIC |
| `*.AIC_` / `*.BIC_` | Critères de sélection de modèle | Choix du meilleur modèle |
| `*.predict(t)` | Prédiction de S(t) paramétrique | Superposition sur courbe KM |
""")

    st.markdown("##### Tests statistiques")
    st.markdown("""
| Classe / Fonction | Description | Utilisation dans l'app |
|---|---|---|
| `logrank_test(T1, T2, E1, E2)` | Test du Log-Rank (Mantel-Haenszel) pour 2 groupes | Comparaison de courbes par variable |
| `multivariate_logrank_test(T, group, E)` | Log-Rank pour > 2 groupes | Test global multi-groupes |
| `proportional_hazard_test(cph, df)` | Test de Schoenfeld (proportionnalité) | Vérification des hypothèses de Cox |
""")

    # ══════════════════════════════════════════════════════════════════════════
    # 3. ARCHITECTURE DE L'APPLICATION
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### 3. Architecture de l'application")

    st.code("""
pid/
|-- app.py                    # Point d'entrée Streamlit
|-- requirements.txt          # Dépendances Python
|-- .streamlit/config.toml    # Thème et configuration
|-- data/
|   |-- survival_data_1000.csv  # Jeu de données
|-- tabs/                     # Modules des onglets
|   |-- data_viz.py           # Visualisation des données
|   |-- missing_data.py       # Gestion des manquantes
|   |-- descriptive.py        # Statistiques descriptives
|   |-- charts.py             # Représentations graphiques
|   |-- survival.py           # Kaplan-Meier & Nelson-Aalen
|   |-- prediction.py         # Prédiction individuelle
|   |-- cox_model.py          # Modèle de Cox
|   |-- comorbidities.py      # Analyse des comorbidités
|   |-- advanced.py           # Analyses avancées
|   |-- comparator.py         # Comparateur de sous-groupes
|   |-- export.py             # Export des résultats
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
    st.markdown("#### 4. Flux de données")

    st.code("""
CSV (upload ou défaut)
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
  |- Paramétriques (Weibull, LogNormal, LogLogistic)
  |- Tests (logrank_test, proportional_hazard_test)
    """, language="text")

    # ══════════════════════════════════════════════════════════════════════════
    # 5. ONGLETS
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### 5. Description des onglets")

    onglets = [
        ("Données", "Vue d'ensemble du dataset : KPIs (patients, variables, événements, taux de censure), "
                     "distribution du temps de suivi, aperçu des données, inspection des types."),
        ("Manquantes", "Détection et traitement des valeurs manquantes : tableau, visualisation, "
                        "stratégies de remplacement (suppression, moyenne, médiane, mode)."),
        ("Statistiques", "Statistiques descriptives : variables quantitatives (describe), "
                          "variables qualitatives (effectifs, fréquences), variables dérivées."),
        ("Graphiques", "Exploration visuelle : histogrammes, boxplots, bar charts, pie charts, "
                        "scatter plots avec coloration par variable catégorielle."),
        ("Survie", "Analyse de survie : Kaplan-Meier (globale, stratifiée, tableau des survivants), "
                    "Nelson-Aalen (risque cumulé, estimation interactive), tests Log-Rank."),
        ("Prédiction", "Prédiction individuelle : saisie du profil patient, courbe de survie prédite "
                        "par le modèle de Cox, comparaison avec profils de référence."),
        ("Modèle de Cox", "Régression de Cox : tableau des coefficients et Hazard Ratios, "
                           "forest plot, courbes ajustées, test de Schoenfeld."),
        ("Comorbidités", "Analyse dédiée : distribution, impact sur la survie (KM + Log-Rank), "
                          "profil par groupe, interaction avec le traitement."),
        ("Avancé", "Modèles paramétriques (Weibull, Log-Normal, Log-Logistique) avec AIC/BIC, "
                    "matrice de corrélation, résidus de Cox, analyse de sensibilité."),
        ("Comparateur", "Comparaison côte à côte de deux sous-populations définies par l'utilisateur : "
                         "courbes KM, test Log-Rank, métriques, profil démographique, prédiction Cox."),
        ("Export", "Téléchargement des résultats : données filtrées, statistiques descriptives, "
                    "table de survie, métriques, résultats Cox (CSV et Excel)."),
    ]

    for name, desc in onglets:
        st.markdown(f"**{name}** - {desc}")
