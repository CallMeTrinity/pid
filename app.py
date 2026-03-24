import streamlit as st
import pandas as pd
from pathlib import Path
from utils.data_loader import load_csv, process_data, handle_missing, apply_filters

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Analyse de Survie",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Analyse de Survie des Patients")
st.caption("Master MIAGE M1 — Data Science et Applications (2025-2026)")

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # ── Data loading ──────────────────────────────────────────────────────────
    st.header("Chargement des donnees")

    uploaded_file = st.file_uploader("Fichier CSV", type=["csv"])
    encoding = st.selectbox("Encodage", ["utf-8", "latin-1", "cp1252", "utf-16"])
    separator = st.selectbox("Separateur", [",", ";", "\\t", "|"])

    sep_char = "\t" if separator == "\\t" else separator

    default_path = Path(__file__).parent / "data" / "survival_data_1000.csv"

    if uploaded_file is not None:
        df_raw = load_csv(uploaded_file, encoding, sep_char)
    elif default_path.exists():
        df_raw = load_csv(str(default_path), "utf-8", ",")
    else:
        st.error("Aucun fichier charge. Deposez un CSV ou placez-le dans `data/`.")
        st.stop()

    # ── Missing data (applied from tab config) ────────────────────────────────
    if "df_clean" in st.session_state:
        df_raw = st.session_state["df_clean"]

    # ── Column selection ──────────────────────────────────────────────────────
    st.markdown("---")
    st.header("Variables d'analyse")
    all_cols = list(df_raw.columns)

    time_col = st.selectbox(
        "Variable temps (duree)",
        all_cols,
        index=all_cols.index("Time_to_Event") if "Time_to_Event" in all_cols else 0,
    )
    event_col = st.selectbox(
        "Variable evenement",
        all_cols,
        index=all_cols.index("Event_Observed") if "Event_Observed" in all_cols else 0,
    )

    # ── Process ───────────────────────────────────────────────────────────────
    df = process_data(df_raw, time_col, event_col)

    # ── Filters ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.header("Filtres")
    filters = {}

    if "Age" in df.columns:
        age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
        filters["Age"] = st.slider("Age", age_min, age_max, (age_min, age_max))

    if "Sex" in df.columns:
        opts = sorted(df["Sex"].dropna().unique().tolist())
        filters["Sex"] = st.multiselect("Sexe", opts, default=opts)

    if "Smoker" in df.columns:
        opts = sorted(df["Smoker"].dropna().unique().tolist())
        filters["Smoker"] = st.multiselect("Fumeur", opts, default=opts,
                                           format_func=lambda x: f"{'Oui' if x == 1 else 'Non'} ({x})")

    if "Treatment" in df.columns:
        opts = sorted(df["Treatment"].dropna().unique().tolist())
        filters["Treatment"] = st.multiselect("Traitement", opts, default=opts)

    if "Physical_Activity" in df.columns:
        opts = sorted(df["Physical_Activity"].dropna().unique().tolist())
        filters["Physical_Activity"] = st.multiselect("Activite physique", opts, default=opts)

    if "BMI" in df.columns:
        bmin, bmax = float(df["BMI"].min()), float(df["BMI"].max())
        filters["BMI"] = st.slider("IMC (BMI)", bmin, bmax, (bmin, bmax))

    if "Comorbidities" in df.columns:
        cmin, cmax = int(df["Comorbidities"].min()), int(df["Comorbidities"].max())
        filters["Comorbidities"] = st.slider("Comorbidites", cmin, cmax, (cmin, cmax))

    # Apply filters
    df_filtered = apply_filters(df, filters)

    st.markdown("---")
    st.metric("Patients (filtres)", f"{len(df_filtered)} / {len(df)}")

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
from tabs.data_viz import render as tab_data_viz
from tabs.missing_data import render as tab_missing
from tabs.descriptive import render as tab_descriptive
from tabs.charts import render as tab_charts
from tabs.survival import render as tab_survival
from tabs.prediction import render as tab_prediction
from tabs.cox_model import render as tab_cox

tab_names = [
    "Visualisation",
    "Donnees manquantes",
    "Statistiques",
    "Graphiques",
    "Survie",
    "Prediction",
    "Modele de Cox",
]

tabs = st.tabs(tab_names)

with tabs[0]:
    tab_data_viz(df_filtered, time_col, event_col)
with tabs[1]:
    tab_missing(df_raw, df)
with tabs[2]:
    tab_descriptive(df_filtered, time_col, event_col)
with tabs[3]:
    tab_charts(df_filtered, time_col, event_col)
with tabs[4]:
    tab_survival(df_filtered, time_col, event_col)
with tabs[5]:
    tab_prediction(df, time_col, event_col)
with tabs[6]:
    tab_cox(df_filtered, time_col, event_col)
