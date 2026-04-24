import streamlit as st
import pandas as pd
from pathlib import Path
from utils.data_loader import load_csv, process_data, handle_missing, apply_filters

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Analyse de Survie",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ─────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body {
    font-family: 'Inter', sans-serif;
}

/* ── Hero header ────────────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #6C63FF 0%, #3B82F6 50%, #06B6D4 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    color: white;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 300px;
    height: 300px;
    background: rgba(255,255,255,0.05);
    border-radius: 50%;
}
.hero h1 {
    margin: 0;
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.5px;
}
.hero p {
    margin: 0.5rem 0 0 0;
    font-size: 0.95rem;
    opacity: 0.9;
}

/* ── Metric cards ───────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(59,130,246,0.08), rgba(6,182,212,0.08));
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricLabel"] {
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    opacity: 0.7;
}
[data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}

/* ── Tabs ────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: rgba(108,99,255,0.05);
    border-radius: 12px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 0.6rem 1.2rem;
    font-weight: 500;
    font-size: 0.85rem;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6C63FF, #3B82F6) !important;
    color: white !important;
    border-radius: 8px !important;
}

/* ── Sidebar ─────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0E1117 0%, #131720 100%);
}
section[data-testid="stSidebar"] [data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(6,182,212,0.1), rgba(108,99,255,0.1));
    border-color: rgba(6,182,212,0.3);
}

/* ── DataFrames ──────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

/* ── Expanders ───────────────────────────────────────────────── */
.streamlit-expanderHeader {
    font-weight: 600;
    font-size: 0.95rem;
}

/* ── Buttons ─────────────────────────────────────────────────── */
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
}

/* ── Section dividers ────────────────────────────────────────── */
.section-divider {
    height: 2px;
    background: linear-gradient(90deg, #3B82F6, #06B6D4, transparent);
    border: none;
    border-radius: 2px;
    margin: 1.5rem 0;
    opacity: 0.6;
}

/* ── Info cards ──────────────────────────────────────────────── */
.info-card {
    background: rgba(108,99,255,0.06);
    border-left: 4px solid #6C63FF;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.5rem;
    margin: 1rem 0;
}
.info-card h4 {
    margin: 0 0 0.3rem 0;
    color: #6C63FF;
}
.info-card p {
    margin: 0;
    font-size: 0.9rem;
    opacity: 0.85;
}

/* ── Sidebar logo area ───────────────────────────────────────── */
.sidebar-header {
    text-align: center;
    padding: 0.5rem 0 1rem 0;
    border-bottom: 1px solid rgba(59,130,246,0.2);
    margin-bottom: 1rem;
}
.sidebar-header h2 {
    font-size: 1.1rem;
    margin: 0;
    background: linear-gradient(135deg, #3B82F6, #06B6D4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* ── KPI row ─────────────────────────────────────────────────── */
.kpi-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Hero header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>Analyse de Survie des Patients</h1>
    <p>Master MIAGE M1 — Projet Ingénierie de Données (2025-2026) · Pipeline interactif d'analyse de survie</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <h2>SurvivalLab</h2>
    </div>
    """, unsafe_allow_html=True)


    # ── Data loading ──────────────────────────────────────────────────────────
    with st.expander("Chargement des donnees", expanded=True):
        uploaded_file = st.file_uploader("Fichier CSV", type=["csv"], label_visibility="collapsed")
        col1, col2 = st.columns(2)
        with col1:
            encoding = st.selectbox("Encodage", ["utf-8", "latin-1", "cp1252", "utf-16"],
                                    label_visibility="collapsed",
                                    help="Encodage du fichier CSV")
        with col2:
            separator = st.selectbox("Separateur", [",", ";", "\\t", "|"],
                                     label_visibility="collapsed",
                                     help="Separateur de colonnes")

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
    with st.expander("Variables d'analyse", expanded=True):
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
    # Handle reset before widgets are instantiated
    if st.session_state.pop("_reset_filters", False):
        for k in ("flt_age", "flt_sex", "flt_smoker", "flt_treatment",
                  "flt_activity", "flt_bmi", "flt_comorb"):
            st.session_state.pop(k, None)

    with st.expander("Filtres", expanded=True):
        if st.button("Reinitialiser les filtres", use_container_width=True, key="reset_filters_btn"):
            st.session_state["_reset_filters"] = True
            st.rerun()

        filters = {}
        ACTIVITY_ORDER = ["Low", "Moderate", "High"]

        if "Age" in df.columns:
            age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
            filters["Age"] = st.slider("Age", age_min, age_max,
                                       (age_min, age_max), key="flt_age")

        if "Sex" in df.columns:
            opts = sorted(df["Sex"].dropna().unique().tolist())
            filters["Sex"] = st.multiselect("Sexe", opts, default=opts, key="flt_sex")

        if "Smoker" in df.columns:
            opts = sorted(df["Smoker"].dropna().unique().tolist())
            filters["Smoker"] = st.multiselect(
                "Fumeur", opts, default=opts, key="flt_smoker",
                format_func=lambda x: "Oui" if x == 1 else "Non",
            )

        if "Treatment" in df.columns:
            opts = sorted(df["Treatment"].dropna().unique().tolist())
            filters["Treatment"] = st.multiselect(
                "Traitement", opts, default=opts, key="flt_treatment")

        if "Physical_Activity" in df.columns:
            raw_opts = df["Physical_Activity"].dropna().unique().tolist()
            opts = [a for a in ACTIVITY_ORDER if a in raw_opts] + \
                   [o for o in raw_opts if o not in ACTIVITY_ORDER]
            filters["Physical_Activity"] = st.multiselect(
                "Activite physique", opts, default=opts, key="flt_activity")

        if "BMI" in df.columns:
            bmin, bmax = int(df["BMI"].min()), int(df["BMI"].max())
            filters["BMI"] = st.slider("IMC (BMI)", bmin, bmax,
                                       (bmin, bmax), key="flt_bmi")

        if "Comorbidities" in df.columns:
            cmin, cmax = int(df["Comorbidities"].min()), int(df["Comorbidities"].max())
            filters["Comorbidities"] = st.slider(
                "Comorbidites", cmin, cmax, (cmin, cmax), key="flt_comorb")

    # Apply filters
    df_filtered = apply_filters(df, filters)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Patient count ─────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    col1.metric("Patients", f"{len(df_filtered)}")
    col2.metric("Evenements", f"{int(df_filtered[event_col].sum())}")

    pct = len(df_filtered) / len(df) * 100 if len(df) > 0 else 0
    st.progress(pct / 100, text=f"{pct:.0f}% du dataset ({len(df_filtered)}/{len(df)})")

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
from tabs.comorbidities import render as tab_comorbidities
from tabs.advanced import render as tab_advanced
from tabs.comparator import render as tab_comparator
from tabs.export import render as tab_export
from tabs.about import render as tab_about

tab_names = [
    "Donnees",
    "Manquantes",
    "Statistiques",
    "Graphiques",
    "Survie",
    "Prediction",
    "Modele de Cox",
    "Comparateur",
    "Plus d'infos",
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
with tabs[7]:
    tab_comparator(df_filtered, time_col, event_col)
with tabs[8]:
    sub_names = ["Comorbidites", "Avance", "Export", "A propos"]
    sub_tabs = st.tabs(sub_names)
    with sub_tabs[0]:
        tab_comorbidities(df_filtered, time_col, event_col)
    with sub_tabs[1]:
        tab_advanced(df_filtered, time_col, event_col)
    with sub_tabs[2]:
        tab_export(df_filtered, time_col, event_col)
    with sub_tabs[3]:
        tab_about()
