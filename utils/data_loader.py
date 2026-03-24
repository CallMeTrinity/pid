import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

DEFAULT_DATA = Path(__file__).parent.parent / "data" / "survival_data_1000.csv"


@st.cache_data(show_spinner=False)
def load_csv(source, encoding="utf-8", separator=","):
    """Load a CSV from a file path or UploadedFile."""
    try:
        return pd.read_csv(source, encoding=encoding, sep=separator)
    except Exception as e:
        st.error(f"Erreur de lecture : {e}")
        st.stop()


def process_data(df: pd.DataFrame, time_col: str, event_col: str) -> pd.DataFrame:
    """Create derived variables and remove duplicates."""
    df = df.copy()

    # Tranche d'age
    if "Age" in df.columns:
        df["Tranche_Age"] = pd.cut(
            df["Age"], bins=[0, 50, 60, np.inf],
            labels=["<50", "50-60", ">60"], right=True,
        )

    # Tranche IMC
    if "BMI" in df.columns:
        df["Tranche_BMI"] = pd.cut(
            df["BMI"], bins=[0, 18, 26, np.inf],
            labels=["<18", "18-26", ">26"], right=True,
        )

    # Suppression des vrais doublons (lignes identiques sur toutes les colonnes d'origine)
    # Si un patient apparait 2 fois, garder la ligne avec Event=1
    orig_cols = [c for c in df.columns if c not in ["Tranche_Age", "Tranche_BMI"]]
    before = len(df)
    df = df.sort_values(event_col, ascending=False).drop_duplicates(subset=orig_cols, keep="first")
    removed = before - len(df)
    if removed > 0:
        st.sidebar.info(f"{removed} doublon(s) exact(s) supprime(s)")

    return df.reset_index(drop=True)


def handle_missing(df: pd.DataFrame, strategy: str, cols: list) -> pd.DataFrame:
    """Apply a missing-data strategy to selected columns."""
    df = df.copy()
    if strategy == "drop_rows":
        df = df.dropna(subset=cols)
    elif strategy == "drop_cols":
        df = df.drop(columns=[c for c in cols if df[c].isna().any()])
    elif strategy == "mean":
        for c in cols:
            if pd.api.types.is_numeric_dtype(df[c]):
                df[c] = df[c].fillna(df[c].mean())
    elif strategy == "median":
        for c in cols:
            if pd.api.types.is_numeric_dtype(df[c]):
                df[c] = df[c].fillna(df[c].median())
    elif strategy == "mode":
        for c in cols:
            if not df[c].mode().empty:
                df[c] = df[c].fillna(df[c].mode().iloc[0])
    return df


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply sidebar filters to the DataFrame."""
    df = df.copy()
    for col, val in filters.items():
        if col not in df.columns:
            continue
        if isinstance(val, tuple) and len(val) == 2:
            df = df[(df[col] >= val[0]) & (df[col] <= val[1])]
        elif isinstance(val, list):
            df = df[df[col].isin(val)]
    return df


@st.cache_resource(show_spinner="Ajustement du modele de Cox...")
def fit_cox_model(_df_hash: str, cox_data: pd.DataFrame, time_col: str, event_col: str):
    """Fit and cache a Cox PH model. Drops constant columns to avoid singularity."""
    from lifelines import CoxPHFitter

    feature_cols = [c for c in cox_data.columns if c not in [time_col, event_col]]
    # Remove constant columns (zero variance) — caused by filters narrowing data
    varying = [c for c in feature_cols if cox_data[c].nunique() > 1]
    dropped = set(feature_cols) - set(varying)

    cox_data = cox_data[varying + [time_col, event_col]]

    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(cox_data, duration_col=time_col, event_col=event_col)
    return cph, dropped


def prepare_cox_data(df: pd.DataFrame, time_col: str, event_col: str) -> pd.DataFrame:
    """Encode categorical variables for Cox model."""
    cox = df.copy()

    if "Sex" in cox.columns:
        cox["Sex_Female"] = (cox["Sex"] == "Female").astype(int)
    if "Treatment" in cox.columns:
        cox["Treatment_Experimental"] = (cox["Treatment"] == "Experimental").astype(int)
    if "Physical_Activity" in cox.columns:
        dummies = pd.get_dummies(cox["Physical_Activity"], prefix="Activity")
        if "Activity_Low" in dummies.columns:
            dummies = dummies.drop(columns=["Activity_Low"])
        cox = pd.concat([cox, dummies], axis=1)

    features = [c for c in [
        "Age", "Sex_Female", "Smoker",
        "Treatment_Experimental", "Activity_High", "Activity_Moderate",
    ] if c in cox.columns]

    return cox[features + [time_col, event_col]]
