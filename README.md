# Analyse de Survie des Patients

**Master MIAGE M1 — Projet Ingenierie de Donnees (2025-2026)**

Application web interactive d'analyse de survie construite avec Streamlit et la librairie lifelines. Elle permet d'explorer un jeu de donnees de patients, d'estimer les probabilites de survie, d'identifier les facteurs de risque et de comparer des sous-populations.

---

## Sommaire

- [Contexte](#contexte)
- [Fonctionnalites](#fonctionnalites)
- [Donnees](#donnees)
- [Installation](#installation)
- [Lancement](#lancement)
- [Architecture du projet](#architecture-du-projet)
- [Bibliotheques utilisees](#bibliotheques-utilisees)
- [Description des onglets](#description-des-onglets)
- [Deploiement](#deploiement)
- [Difficultes rencontrees](#difficultes-rencontrees)
- [Pistes d'amelioration](#pistes-damelioration)

---

## Contexte

L'analyse de survie etudie le temps ecoule avant la survenue d'un evenement (deces, panne, rechute, etc.), en tenant compte des donnees **censurees** — des individus pour lesquels l'evenement n'a pas encore ete observe a la fin du suivi.

Ce projet applique les methodes classiques d'analyse de survie (Kaplan-Meier, Nelson-Aalen, regression de Cox, modeles parametriques) a un jeu de donnees de 1000 patients afin d'identifier les facteurs influencant significativement la duree de survie.

---

## Fonctionnalites

### Lecture et preparation des donnees
- Chargement de fichiers CSV via upload ou fichier par defaut
- Choix de l'encodage (UTF-8, Latin-1, CP1252, UTF-16) et du separateur
- Selection dynamique des variables temps et evenement
- Creation de variables derivees : `Tranche_Age` (<50, 50-60, >60) et `Tranche_BMI` (<18, 18-26, >26)
- Detection et suppression des doublons
- Gestion des donnees manquantes (suppression, imputation par moyenne/mediane/mode)

### Filtrage interactif
Barre laterale avec filtres sur toutes les variables : Age, Sexe, Fumeur, Traitement, Activite physique, IMC, Comorbidites. Le nombre de patients selectionnes est affiche en temps reel.

### Analyses statistiques
- **Kaplan-Meier** : courbes de survie globales et stratifiees, tableau des proportions de survivants, intervalles de confiance
- **Nelson-Aalen** : risque cumule, estimation interactive pour un temps saisi
- **Modele de Cox** : Hazard Ratios, intervalles de confiance, test de Schoenfeld, courbes ajustees, forest plot
- **Tests Log-Rank** : comparaison de deux ou plusieurs groupes (Mantel-Haenszel)
- **Modeles parametriques** : Weibull, Log-Normal, Log-Logistique avec comparaison AIC/BIC
- **Analyse des residus** : martingale, deviance, Schoenfeld
- **Analyse de sensibilite** : robustesse des resultats selon differents scenarios d'exclusion
- **Analyse des comorbidites** : impact sur la survie, profil par groupe, interaction avec le traitement

### Prediction
Saisie d'un profil patient (age, sexe, fumeur, traitement, activite) et affichage de la courbe de survie predite par le modele de Cox, avec comparaison a des profils de reference.

### Comparateur de sous-groupes
Definition de deux sous-populations avec des filtres independants, comparaison des courbes KM, test du Log-Rank, metriques comparees et prediction Cox du profil moyen.

### Export
Telechargement des resultats au format CSV ou Excel (donnees filtrees, statistiques, table de survie, resultats Cox).

---

## Donnees

Le fichier `data/survival_data_1000.csv` contient 1000 observations avec les variables suivantes :

| Variable | Type | Description |
|---|---|---|
| `Age` | Numerique | Age du patient (annees) |
| `Sex` | Categorielle | Male / Female |
| `Smoker` | Binaire | 0 = Non-fumeur, 1 = Fumeur |
| `Comorbidities` | Numerique | Nombre de comorbidites (0 a 3) |
| `Treatment` | Categorielle | Standard / Experimental |
| `BMI` | Numerique | Indice de masse corporelle |
| `Physical_Activity` | Categorielle | Low / Moderate / High |
| `Time_to_Event` | Numerique | Duree de suivi (mois) |
| `Event_Observed` | Binaire | 0 = Censure, 1 = Deces |

---

## Installation

### Prerequis
- Python 3.10 ou superieur

### Etapes

```bash
# Cloner le depot
git clone <url-du-depot>
cd pid

# Creer un environnement virtuel (recommande)
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows

# Installer les dependances
pip install -r requirements.txt
```

---

## Lancement

```bash
streamlit run app.py
```

L'application s'ouvre dans le navigateur a l'adresse `http://localhost:8501`.

---

## Architecture du projet

```
pid/
├── app.py                       # Point d'entree Streamlit
├── requirements.txt             # Dependances Python
├── README.md
├── .streamlit/
│   └── config.toml              # Theme et configuration Streamlit
├── data/
│   └── survival_data_1000.csv   # Jeu de donnees
├── tabs/                        # Modules des onglets
│   ├── data_viz.py              # Vue d'ensemble des donnees
│   ├── missing_data.py          # Gestion des manquantes
│   ├── descriptive.py           # Statistiques descriptives
│   ├── charts.py                # Representations graphiques
│   ├── survival.py              # Kaplan-Meier, Nelson-Aalen, Log-Rank
│   ├── prediction.py            # Prediction individuelle (Cox)
│   ├── cox_model.py             # Regression de Cox
│   ├── comorbidities.py         # Analyse des comorbidites
│   ├── advanced.py              # Modeles parametriques, residus, sensibilite
│   ├── comparator.py            # Comparateur de sous-groupes
│   ├── export.py                # Export CSV / Excel
│   └── about.py                 # Guide lifelines et documentation
├── utils/
│   ├── data_loader.py           # Chargement, traitement, filtres, modele Cox
│   └── plots.py                 # Fonctions de visualisation (KM, NA, forest plot)
└── docs/
    ├── Projet PID 2025-2026.pdf
    └── Analyse_de_Survie_avec_Python_et_lifelines.ipynb
```

---

## Bibliotheques utilisees

| Bibliotheque | Version min. | Role |
|---|---|---|
| **streamlit** | 1.35.0 | Framework web interactif (interface, widgets, layout) |
| **lifelines** | 0.28.0 | Analyse de survie : Kaplan-Meier, Nelson-Aalen, Cox, modeles parametriques, tests statistiques |
| **pandas** | 2.0.0 | Manipulation de donnees (chargement CSV, transformations, aggregations) |
| **numpy** | 1.24.0 | Calcul numerique |
| **plotly** | 5.18.0 | Graphiques interactifs (histogrammes, scatter, heatmaps, courbes de survie) |
| **matplotlib** | 3.7.0 | Graphiques statiques (courbes KM, forest plot) |
| **seaborn** | 0.12.0 | Visualisation statistique |
| **openpyxl** | 3.1.0 | Export des resultats au format Excel (.xlsx) |

---

## Description des onglets

L'application est organisee en **9 onglets principaux**. Les fonctionnalites secondaires (analyses poussees, export, documentation) sont regroupees dans l'onglet **Plus d'infos** sous forme de sous-onglets.

### Onglets principaux

#### Donnees
Vue d'ensemble : nombre de patients, variables, evenements, taux de censure. Distribution du temps de suivi, apercu du dataset, inspection des types de variables, verification des doublons.

#### Manquantes
Detection des valeurs manquantes avec visualisation. Strategies de traitement : suppression de lignes/colonnes, remplacement par la moyenne, mediane ou mode.

#### Statistiques
Statistiques descriptives des variables quantitatives (moyenne, mediane, ecart-type, quartiles) et qualitatives (effectifs, frequences). Variables derivees (tranches d'age et d'IMC).

#### Graphiques
Exploration visuelle interactive : histogrammes, boxplots, bar charts, pie charts pour les variables qualitatives et quantitatives. Scatter plot avec coloration par variable categorielle.

#### Survie
- **Kaplan-Meier** : courbe globale avec IC, tableau complet des proportions de survivants, courbes stratifiees par variable, survie mediane par groupe.
- **Nelson-Aalen** : risque cumule global et stratifie, estimation interactive S(t) pour un temps saisi.
- **Tests de comparaison** : Log-Rank pour toutes les variables, comparaison detaillee.

#### Prediction
Saisie interactive d'un profil patient. Courbe de survie predite par le modele de Cox. Probabilites de survie a 12, 24, 36, 60 et 100 mois. Comparaison avec des profils de reference (haut risque, intermediaire, protege).

#### Modele de Cox
Regression de Cox a risques proportionnels : tableau des coefficients et Hazard Ratios avec IC 95%, interpretation des facteurs de risque et protecteurs, forest plot, courbes de survie ajustees par covariable, test de Schoenfeld pour la proportionnalite.

#### Comparateur
Outil de comparaison de deux sous-populations. Chaque groupe est defini par des filtres independants. Affichage : courbes KM superposees avec IC, test du Log-Rank, tableau de metriques comparees, profil demographique, prediction Cox du profil moyen.

### Onglet "Plus d'infos" (sous-onglets)

#### Comorbidites
Analyse dediee aux comorbidites : distribution, impact sur la survie (KM + Log-Rank), profil des patients par niveau de comorbidite, taux d'evenements, comparaisons pairwise, interaction comorbidites x traitement.

#### Avance
- **Modeles parametriques** : Weibull, Log-Normal, Log-Logistique. Comparaison AIC/BIC, superposition sur la courbe KM.
- **Correlations** : matrice de correlation (heatmap), correlations avec le temps de survie, scatter plots.
- **Residus de Cox** : residus de martingale, deviance (detection d'outliers), Schoenfeld (verification de la proportionnalite).
- **Analyse de sensibilite** : robustesse des resultats en excluant des sous-groupes (fumeurs, ages, comorbidites, par traitement).

#### Export
Telechargement des resultats :
- Donnees filtrees (CSV)
- Statistiques descriptives (CSV)
- Table de survie Kaplan-Meier avec IC (CSV)
- Metriques de survie (CSV)
- Resultats du modele de Cox (CSV)
- Export complet (Excel multi-onglets)

#### A propos
Documentation integree : librairies utilisees, guide complet des fonctions lifelines avec leur utilisation dans l'application, architecture du projet, flux de donnees, description de chaque onglet.

---

## Deploiement

### Streamlit Community Cloud (utilise pour ce projet)

L'application est deployee sur [Streamlit Community Cloud](https://share.streamlit.io) et branchee sur la branche **`production`** du depot GitHub.

Workflow de deploiement :

1. Le developpement se fait sur `main` (et sur les branches personnelles `antonin`, `Juline`, `owen`)
2. Lorsqu'une version est prete a etre publiee, elle est fusionnee dans la branche `production`
3. Streamlit Community Cloud surveille la branche `production` et **redeploie automatiquement** l'application a chaque push
4. L'URL publique pointe toujours sur le dernier commit de `production`, ce qui permet de garder `main` en developpement actif sans impacter la version en ligne

Configuration initiale (deja faite) :

1. Connexion a [share.streamlit.io](https://share.streamlit.io) avec le compte GitHub
2. Selection du depot `CallMeTrinity/pid`, branche `production`, fichier principal `app.py`
3. Les dependances sont installees depuis `requirements.txt`, la configuration visuelle (theme) vient de `.streamlit/config.toml`

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

## Difficultes rencontrees

- **Prise en main de lifelines** : la librairie couvre beaucoup de methodes (Kaplan-Meier, Nelson-Aalen, Cox, modeles parametriques, tests de Schoenfeld, residus) dont les sorties ne sont pas toujours homogenes (DataFrames, objets fittes, dictionnaires). Il a fallu du temps pour comprendre quelle classe renvoie quoi et dans quel format l'injecter dans Plotly/Matplotlib.
- **Etat partage entre onglets** (Streamlit) : la gestion des filtres, du dataset nettoye (`df_clean`) et des variables temps/evenement via `st.session_state` est delicate. Exemple concret : la reinitialisation des filtres qui doit se faire *avant* l'instanciation des widgets, sinon Streamlit leve une exception.
- **Theme clair / sombre** : plusieurs iterations pour que les graphiques Plotly et Matplotlib restent lisibles quel que soit le theme choisi par l'utilisateur (cf. `.streamlit/config.toml` et utilisation du menu natif de Streamlit plutot que du CSS custom).
- **Performance** : avec les filtres multiples et le reajustement du modele de Cox a chaque changement, le temps de reponse peut se degrader. L'usage de `@st.cache_data` sur le chargement et certaines transformations a ete necessaire pour rester fluide.

---

## Pistes d'amelioration

### Pedagogie et interpretation
- **Enrichir les onglets Avance et Comorbidites** avec davantage d'explications pedagogiques : rappeler ce qu'est chaque indicateur (AIC, BIC, residus de martingale, residus de deviance, residus de Schoenfeld), comment le lire, ce qu'un seuil significatif implique concretement pour le patient ou le clinicien. Actuellement ces onglets supposent que l'utilisateur connait deja la theorie ; ajouter des encarts "Comment interpreter ?" les rendrait accessibles a un non-statisticien.
- Ajouter des tooltips sur les metriques affichees dans les tableaux Cox (p-value, HR, IC 95%) pour rappeler leur signification au survol.

### Reorganisation de la navigation (implementee)
La premiere version de l'application affichait **12 onglets** sur une seule ligne, ce qui surchargeait la barre de navigation. La version actuelle regroupe les onglets secondaires dans un onglet **"Plus d'infos"** contenant des **sous-onglets** :

```
Plus d'infos
├── Comorbidites
├── Avance (modeles parametriques, residus, sensibilite)
├── Export
└── A propos
```

On passe ainsi de 12 a 9 onglets principaux (Donnees, Manquantes, Statistiques, Graphiques, Survie, Prediction, Modele de Cox, Comparateur, Plus d'infos), avec une hierarchie claire : analyses principales en haut, outils et analyses poussees dans le sous-menu.

### Fonctionnalites
- **Export PDF** d'un rapport d'analyse complet (graphiques + tableaux + interpretation) pour un profil patient donne.
- **Sauvegarde et rechargement d'une session** (filtres, variables selectionnees, profil de prediction) via un fichier JSON.
- **Comparateur a N groupes** (actuellement limite a 2) pour comparer plusieurs strates simultanement.
- **Modeles complementaires** : random survival forest, Cox avec penalisation (Lasso/Ridge via lifelines.CoxPHFitter.fit(penalizer=...)).
- **Validation croisee** du modele de Cox (C-index, courbes de calibration) pour evaluer la qualite predictive.
- **Gestion de plusieurs datasets** : permettre de charger plusieurs fichiers et de les comparer.

### Technique
- Ajouter des **tests unitaires** sur `utils/data_loader.py` (traitement des doublons, filtres, imputation) avec pytest.
- Migrer les traitements lourds dans un **cache plus agressif** (`@st.cache_resource` pour le modele Cox fitte).
- Typage strict (mypy) sur les modules utilitaires.

---

## Auteurs

- **Antonin Pamart**
- **Juline Busson**
- **Owen Gicquel**

Master MIAGE M1 — Projet Ingenierie de Donnees (2025-2026)
