import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "survival_data_1000.csv"


@st.cache_data
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        st.error(
            "**Fichier de données introuvable.**\n\n"
            f"Veuillez placer `survival_data_1000.csv` dans le dossier `data/`."
        )
        st.stop()

    df = pd.read_csv(DATA_PATH)

    df["Tranche_Age"] = pd.cut(
        df["Age"],
        bins=[0, 50, 60, np.inf],
        labels=["<50", "50-60", ">60"],
        right=True,
    )
    df["Tranche_BMI"] = pd.cut(
        df["BMI"],
        bins=[0, 18, 26, np.inf],
        labels=["<18", "18-26", ">26"],
        right=True,
    )

    return df


@st.cache_data
def prepare_cox_data(df: pd.DataFrame) -> pd.DataFrame:
    cox_df = df.copy()
    cox_df["Sex_Female"] = (cox_df["Sex"] == "Female").astype(int)
    cox_df["Treatment_Experimental"] = (cox_df["Treatment"] == "Experimental").astype(int)

    activity_dummies = pd.get_dummies(cox_df["Physical_Activity"], prefix="Activity")
    if "Activity_Low" in activity_dummies.columns:
        activity_dummies = activity_dummies.drop(columns=["Activity_Low"])

    cox_df = pd.concat([cox_df, activity_dummies], axis=1)

    features = [
        "Age", "Sex_Female", "Smoker", "Treatment_Experimental",
        "Activity_High", "Activity_Moderate",
    ]
    return cox_df[features + ["Time_to_Event", "Event_Observed"]]


@st.cache_resource
def fit_cox_model():
    from lifelines import CoxPHFitter

    df = load_data()
    cox_data = prepare_cox_data(df)
    cph = CoxPHFitter()
    cph.fit(cox_data, duration_col="Time_to_Event", event_col="Event_Observed")
    return cph
