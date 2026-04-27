# Analyse de Survie des Patients

**Master MIAGE M1 - Projet Ingénierie de Données (2025-2026)**

Application web interactive d'analyse de survie construite avec Streamlit et la librairie lifelines. Elle permet d'explorer un jeu de données de patients, d'estimer les probabilités de survie, d'identifier les facteurs de risque et de comparer des sous-populations.

---

## Sommaire

- [Contexte](#contexte)
- [Fonctionnalités](#fonctionnalités)
- [Données](#données)
- [Installation](#installation)
- [Lancement](#lancement)
- [Architecture du projet](#architecture-du-projet)
- [Bibliothèques utilisées](#bibliothèques-utilisées)
- [Description des onglets](#description-des-onglets)
- [Déploiement](#déploiement)
- [Difficultés rencontrées](#difficultés-rencontrées)
- [Pistes d'amélioration](#pistes-damélioration)

---

## Contexte

L'analyse de survie étudie le temps écoulé avant la survenue d'un événement (décès, panne, rechute, etc.), en tenant compte des données **censurées** (des individus pour lesquels l'événement n'a pas encore été observé à la fin du suivi).

Ce projet applique les méthodes classiques d'analyse de survie (Kaplan-Meier, Nelson-Aalen, régression de Cox, modèles paramétriques) à un jeu de données de 1000 patients afin d'identifier les facteurs influençant significativement la durée de survie.

---

## Fonctionnalités

### Lecture et préparation des données
- Chargement de fichiers CSV via upload ou fichier par défaut
- Choix de l'encodage (UTF-8, Latin-1, CP1252, UTF-16) et du séparateur
- Sélection dynamique des variables temps et événement
- Création de variables dérivées : `Tranche_Age` (<50, 50-60, >60) et `Tranche_BMI` (<18, 18-26, >26)
- Détection et suppression des doublons
- Gestion des données manquantes (suppression, imputation par moyenne/médiane/mode)

### Filtrage interactif
Barre latérale avec filtres sur toutes les variables : Age, Sexe, Fumeur, Traitement, Activité physique, IMC, Comorbidités. Le nombre de patients sélectionnés est affiché en temps réel.

### Analyses statistiques
- **Kaplan-Meier** : courbes de survie globales et stratifiées, tableau des proportions de survivants, intervalles de confiance
- **Nelson-Aalen** : risque cumulé, estimation interactive pour un temps saisi
- **Modèle de Cox** : Hazard Ratios, intervalles de confiance, test de Schoenfeld, courbes ajustées, forest plot
- **Tests Log-Rank** : comparaison de deux ou plusieurs groupes (Mantel-Haenszel)
- **Modèles paramétriques** : Weibull, Log-Normal, Log-Logistique avec comparaison AIC/BIC
- **Analyse des résidus** : martingale, déviance, Schoenfeld
- **Analyse de sensibilité** : robustesse des résultats selon différents scénarios d'exclusion
- **Analyse des comorbidités** : impact sur la survie, profil par groupe, interaction avec le traitement

### Prédiction
Saisie d'un profil patient (age, sexe, fumeur, traitement, activité) et affichage de la courbe de survie prédite par le modèle de Cox, avec comparaison à des profils de référence.

### Comparateur de sous-groupes
Définition de deux sous-populations avec des filtres indépendants, comparaison des courbes KM, test du Log-Rank, métriques comparées et prédiction Cox du profil moyen.

### Export
Téléchargement des résultats au format CSV ou Excel (données filtrées, statistiques, table de survie, résultats Cox).

---

## Données

Le fichier `data/survival_data_1000.csv` contient 1000 observations avec les variables suivantes :

| Variable | Type | Description |
|---|---|---|
| `Age` | Numérique | Âge du patient (années) |
| `Sex` | Catégorielle | Male / Female |
| `Smoker` | Binaire | 0 = Non-fumeur, 1 = Fumeur |
| `Comorbidities` | Numérique | Nombre de comorbidités (0 à 3) |
| `Treatment` | Catégorielle | Standard / Experimental |
| `BMI` | Numérique | Indice de masse corporelle |
| `Physical_Activity` | Catégorielle | Low / Moderate / High |
| `Time_to_Event` | Numérique | Durée de suivi (mois) |
| `Event_Observed` | Binaire | 0 = Censuré, 1 = Décès |

---

## Installation

### Prérequis
- Python 3.10 ou supérieur

### Étapes

```bash
# Cloner le dépôt
git clone <url-du-depot>
cd pid

# Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows

# Installer les dépendances
pip install -r requirements.txt
```

---

## Lancement

```bash
streamlit run app.py
```

L'application s'ouvre dans le navigateur à l'adresse `http://localhost:8501`.

---

## Architecture du projet

```
pid/
├── app.py                       # Point d'entrée Streamlit
├── requirements.txt             # Dépendances Python
├── README.md
├── .streamlit/
│   └── config.toml              # Thème et configuration Streamlit
├── data/
│   └── survival_data_1000.csv   # Jeu de données
├── tabs/                        # Modules des onglets
│   ├── data_viz.py              # Vue d'ensemble des données
│   ├── missing_data.py          # Gestion des manquantes
│   ├── descriptive.py           # Statistiques descriptives
│   ├── charts.py                # Représentations graphiques
│   ├── survival.py              # Kaplan-Meier, Nelson-Aalen, Log-Rank
│   ├── prediction.py            # Prédiction individuelle (Cox)
│   ├── cox_model.py             # Régression de Cox
│   ├── comorbidities.py         # Analyse des comorbidités
│   ├── advanced.py              # Modèles paramétriques, résidus, sensibilité
│   ├── comparator.py            # Comparateur de sous-groupes
│   ├── export.py                # Export CSV / Excel
│   └── about.py                 # Guide lifelines et documentation
├── utils/
│   ├── data_loader.py           # Chargement, traitement, filtres, modèle Cox
│   └── plots.py                 # Fonctions de visualisation (KM, NA, forest plot)
└── docs/
    ├── Projet PID 2025-2026.pdf
    └── Analyse_de_Survie_avec_Python_et_lifelines.ipynb
```

---

## Bibliothèques utilisées

| Bibliothèque | Version min. | Rôle |
|---|---|---|
| **streamlit** | 1.35.0 | Framework web interactif (interface, widgets, layout) |
| **lifelines** | 0.28.0 | Analyse de survie : Kaplan-Meier, Nelson-Aalen, Cox, modèles paramétriques, tests statistiques |
| **pandas** | 2.0.0 | Manipulation de données (chargement CSV, transformations, agrégations) |
| **numpy** | 1.24.0 | Calcul numérique |
| **plotly** | 5.18.0 | Graphiques interactifs (histogrammes, scatter, heatmaps, courbes de survie) |
| **matplotlib** | 3.7.0 | Graphiques statiques (courbes KM, forest plot) |
| **seaborn** | 0.12.0 | Visualisation statistique |
| **openpyxl** | 3.1.0 | Export des résultats au format Excel (.xlsx) |

---

## Description des onglets

L'application est organisée en **9 onglets principaux**. Les fonctionnalités secondaires (analyses poussées, export, documentation) sont regroupées dans l'onglet **Plus d'infos** sous forme de sous-onglets.

### Onglets principaux

#### Données
Vue d'ensemble : nombre de patients, variables, événements, taux de censure. Distribution du temps de suivi, aperçu du dataset, inspection des types de variables, vérification des doublons.

#### Manquantes
Détection des valeurs manquantes avec visualisation. Stratégies de traitement : suppression de lignes/colonnes, remplacement par la moyenne, médiane ou mode.

#### Statistiques
Statistiques descriptives des variables quantitatives (moyenne, médiane, écart-type, quartiles) et qualitatives (effectifs, fréquences). Variables dérivées (tranches d'âge et d'IMC).

#### Graphiques
Exploration visuelle interactive : histogrammes, boxplots, bar charts, pie charts pour les variables qualitatives et quantitatives. Scatter plot avec coloration par variable catégorielle.

#### Survie
- **Kaplan-Meier** : courbe globale avec IC, tableau complet des proportions de survivants, courbes stratifiées par variable, survie médiane par groupe.
- **Nelson-Aalen** : risque cumulé global et stratifié, estimation interactive S(t) pour un temps saisi.
- **Tests de comparaison** : Log-Rank pour toutes les variables, comparaison détaillée.

#### Prédiction
Saisie interactive d'un profil patient. Courbe de survie prédite par le modèle de Cox. Probabilités de survie à 12, 24, 36, 60 et 100 mois. Comparaison avec des profils de référence (haut risque, intermédiaire, protégé).

#### Modèle de Cox
Régression de Cox à risques proportionnels : tableau des coefficients et Hazard Ratios avec IC 95%, interprétation des facteurs de risque et protecteurs, forest plot, courbes de survie ajustées par covariable, test de Schoenfeld pour la proportionnalité.

#### Comparateur
Outil de comparaison de deux sous-populations. Chaque groupe est défini par des filtres indépendants. Affichage : courbes KM superposées avec IC, test du Log-Rank, tableau de métriques comparées, profil démographique, prédiction Cox du profil moyen.

### Onglet "Plus d'infos" (sous-onglets)

#### Comorbidités
Analyse dédiée aux comorbidités : distribution, impact sur la survie (KM + Log-Rank), profil des patients par niveau de comorbidité, taux d'événements, comparaisons pairwise, interaction comorbidités x traitement.

#### Avancé
- **Modèles paramétriques** : Weibull, Log-Normal, Log-Logistique. Comparaison AIC/BIC, superposition sur la courbe KM.
- **Corrélations** : matrice de corrélation (heatmap), corrélations avec le temps de survie, scatter plots.
- **Résidus de Cox** : résidus de martingale, déviance (détection d'outliers), Schoenfeld (vérification de la proportionnalité).
- **Analyse de sensibilité** : robustesse des résultats en excluant des sous-groupes (fumeurs, âgés, comorbidités, par traitement).

#### Export
Téléchargement des résultats :
- Données filtrées (CSV)
- Statistiques descriptives (CSV)
- Table de survie Kaplan-Meier avec IC (CSV)
- Métriques de survie (CSV)
- Résultats du modèle de Cox (CSV)
- Export complet (Excel multi-onglets)

#### À propos
Documentation intégrée : librairies utilisées, guide complet des fonctions lifelines avec leur utilisation dans l'application, architecture du projet, flux de données, description de chaque onglet.

---

## Déploiement

### Streamlit Community Cloud (utilisé pour ce projet)

L'application est déployée sur [Streamlit Community Cloud](https://share.streamlit.io) et branchée sur la branche **`production`** du dépôt GitHub.

Workflow de déploiement :

1. Le développement se fait sur `main` (et sur les branches personnelles `antonin`, `Juline`, `owen`)
2. Lorsqu'une version est prête à être publiée, elle est fusionnée dans la branche `production`
3. Streamlit Community Cloud surveille la branche `production` et **redéploie automatiquement** l'application à chaque push
4. L'URL publique pointe toujours sur le dernier commit de `production`, ce qui permet de garder `main` en développement actif sans impacter la version en ligne

Configuration initiale (déjà faite) :

1. Connexion à [share.streamlit.io](https://share.streamlit.io) avec le compte GitHub
2. Sélection du dépôt `CallMeTrinity/pid`, branche `production`, fichier principal `app.py`
3. Les dépendances sont installées depuis `requirements.txt`, la configuration visuelle (thème) vient de `.streamlit/config.toml`

Lien public : [streamlit cloud](https://ssjafs4uez8nwknknl2yru.streamlit.app/)

### Docker (alternative locale)

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t analyse-survie .
docker run -p 8501:8501 analyse-survie
```

---

## Difficultés rencontrées

- **Prise en main de lifelines** : la librairie couvre beaucoup de méthodes (Kaplan-Meier, Nelson-Aalen, Cox, modèles paramétriques, tests de Schoenfeld, résidus) dont les sorties ne sont pas toujours homogènes (DataFrames, objets fittés, dictionnaires). Il a fallu du temps pour comprendre quelle classe renvoie quoi et dans quel format l'injecter dans Plotly/Matplotlib.
- **État partagé entre onglets** (Streamlit) : la gestion des filtres, du dataset nettoyé (`df_clean`) et des variables temps/événement via `st.session_state` est délicate. Exemple concret : la réinitialisation des filtres qui doit se faire *avant* l'instanciation des widgets, sinon Streamlit lève une exception.
- **Thème clair / sombre** : plusieurs itérations pour que les graphiques Plotly et Matplotlib restent lisibles quel que soit le thème choisi par l'utilisateur (cf. `.streamlit/config.toml` et utilisation du menu natif de Streamlit plutôt que du CSS custom).
- **Performance** : avec les filtres multiples et le réajustement du modèle de Cox à chaque changement, le temps de réponse peut se dégrader. L'usage de `@st.cache_data` sur le chargement et certaines transformations a été nécessaire pour rester fluide.

---

## Pistes d'amélioration

### Pédagogie et interprétation
- **Enrichir les onglets Avancé et Comorbidités** avec davantage d'explications pédagogiques : rappeler ce qu'est chaque indicateur (AIC, BIC, résidus de martingale, résidus de déviance, résidus de Schoenfeld), comment le lire, ce qu'un seuil significatif implique concrètement pour le patient ou le clinicien. Actuellement ces onglets supposent que l'utilisateur connaît déjà la théorie ; ajouter des encarts "Comment interpréter ?" les rendrait accessibles à un non-statisticien.
- Ajouter des tooltips sur les métriques affichées dans les tableaux Cox (p-value, HR, IC 95%) pour rappeler leur signification au survol.

### Réorganisation de la navigation (implémentée)
La première version de l'application affichait **12 onglets** sur une seule ligne, ce qui surchargeait la barre de navigation. La version actuelle regroupe les onglets secondaires dans un onglet **"Plus d'infos"** contenant des **sous-onglets** :

```
Plus d'infos
├── Comorbidités
├── Avancé (modèles paramétriques, résidus, sensibilité)
├── Export
└── À propos
```

On passe ainsi de 12 à 9 onglets principaux (Données, Manquantes, Statistiques, Graphiques, Survie, Prédiction, Modèle de Cox, Comparateur, Plus d'infos), avec une hiérarchie claire : analyses principales en haut, outils et analyses poussées dans le sous-menu.

### Fonctionnalités
- **Export PDF** d'un rapport d'analyse complet (graphiques + tableaux + interprétation) pour un profil patient donné.
- **Sauvegarde et rechargement d'une session** (filtres, variables sélectionnées, profil de prédiction) via un fichier JSON.
- **Comparateur à N groupes** (actuellement limité à 2) pour comparer plusieurs strates simultanément.
- **Modèles complémentaires** : random survival forest, Cox avec pénalisation (Lasso/Ridge via lifelines.CoxPHFitter.fit(penalizer=...)).
- **Validation croisée** du modèle de Cox (C-index, courbes de calibration) pour évaluer la qualité prédictive.
- **Gestion de plusieurs datasets** : permettre de charger plusieurs fichiers et de les comparer.

### Technique
- Ajouter des **tests unitaires** sur `utils/data_loader.py` (traitement des doublons, filtres, imputation) avec pytest.
- Migrer les traitements lourds dans un **cache plus agressif** (`@st.cache_resource` pour le modèle Cox fitté).
- Typage strict (mypy) sur les modules utilitaires.

---

## Auteurs

- **Antonin Pamart**
- **Juline Busson**
- **Owen Gicquel**

Master MIAGE M1 - Projet Ingénierie de Données (2025-2026)
