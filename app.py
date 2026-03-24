import streamlit as st
from utils.data_loader import load_data

st.set_page_config(
    page_title="Analyse de Survie",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Analyse de Survie des Patients")
st.subheader("Master MIAGE M1 — Data Science et Applications")

st.markdown("""
Cette application présente une analyse complète de la survie de patients à partir de
données cliniques. L'objectif est d'identifier les facteurs influençant la survie
et d'estimer les probabilités de survie en fonction des caractéristiques individuelles.

---

### Méthodes utilisées

| Méthode | Type | Objectif |
|---------|------|---------|
| **Kaplan-Meier** | Non-paramétrique | Estimer la fonction de survie S(t) |
| **Nelson-Aalen** | Non-paramétrique | Estimer la fonction de risque cumulée H(t) |
| **Modèle de Cox** | Semi-paramétrique | Modéliser l'effet des covariables sur le risque |

---

### Variables étudiées

| Variable | Type | Description |
|----------|------|-------------|
| `Age` | Continue | Âge du patient (années) |
| `Sex` | Catégorielle | Sexe (Male / Female) |
| `Smoker` | Binaire | Statut fumeur (0 = Non, 1 = Oui) |
| `Comorbidities` | Entière | Nombre de comorbidités |
| `Treatment` | Catégorielle | Type de traitement (Standard / Experimental) |
| `BMI` | Continue | Indice de masse corporelle |
| `Physical_Activity` | Catégorielle | Activité physique (Low / Moderate / High) |
| `Time_to_Event` | Continue | Durée de suivi (mois) |
| `Event_Observed` | Binaire | Évènement observé (0 = Censuré, 1 = Décès) |

---
""")

df = load_data()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Patients", f"{len(df):,}")
col2.metric(
    "Décès observés",
    f"{int(df['Event_Observed'].sum()):,}",
    f"{df['Event_Observed'].mean()*100:.1f}%",
)
col3.metric(
    "Censurés",
    f"{int((1 - df['Event_Observed']).sum()):,}",
    f"{(1 - df['Event_Observed'].mean())*100:.1f}%",
)
col4.metric("Durée médiane de suivi", f"{df['Time_to_Event'].median():.1f} mois")

st.markdown("---")
st.markdown("### Aperçu des données brutes")
st.dataframe(df.head(10), use_container_width=True)

st.markdown("---")
st.caption("Utilisez le menu de navigation à gauche pour accéder aux différentes sections de l'analyse.")
