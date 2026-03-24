import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import KaplanMeierFitter, NelsonAalenFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

PALETTE = sns.color_palette("husl", 6)
plt.rcParams["font.size"] = 11


# ── Kaplan-Meier ─────────────────────────────────────────────────────────────

def plot_km_global(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    kmf = KaplanMeierFitter()
    kmf.fit(df["Time_to_Event"], event_observed=df["Event_Observed"], label="Population globale")
    kmf.plot_survival_function(ax=ax, ci_show=True, color=PALETTE[0])

    ax.set_title("Courbe de survie globale (Kaplan-Meier)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Temps (mois)")
    ax.set_ylabel("Probabilité de survie S(t)")
    ax.set_ylim(0, 1.05)

    median_surv = kmf.median_survival_time_
    ax.axvline(x=median_surv, color="red", linestyle="--", alpha=0.7, label=f"Médiane = {median_surv:.1f} mois")
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, kmf


def plot_km_stratified(df: pd.DataFrame, group_col: str):
    groups = df[group_col].dropna().unique()
    fig, ax = plt.subplots(figsize=(10, 5))

    kmf = KaplanMeierFitter()
    for i, grp in enumerate(sorted(groups, key=str)):
        mask = df[group_col] == grp
        kmf.fit(
            df.loc[mask, "Time_to_Event"],
            event_observed=df.loc[mask, "Event_Observed"],
            label=str(grp),
        )
        kmf.plot_survival_function(ax=ax, ci_show=True, color=PALETTE[i % len(PALETTE)])

    ax.set_title(f"Courbes de survie par {group_col}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Temps (mois)")
    ax.set_ylabel("Probabilité de survie S(t)")
    ax.set_ylim(0, 1.05)
    ax.legend(title=group_col)
    fig.tight_layout()
    return fig


def logrank_result(df: pd.DataFrame, group_col: str) -> dict:
    groups = df[group_col].dropna().unique()
    if len(groups) == 2:
        g0, g1 = sorted(groups, key=str)
        m0 = df[group_col] == g0
        m1 = df[group_col] == g1
        result = logrank_test(
            df.loc[m0, "Time_to_Event"], df.loc[m1, "Time_to_Event"],
            event_observed_A=df.loc[m0, "Event_Observed"],
            event_observed_B=df.loc[m1, "Event_Observed"],
        )
        return {"test": "Log-rank (2 groupes)", "p_value": result.p_value, "statistic": result.test_statistic}
    else:
        result = multivariate_logrank_test(
            df["Time_to_Event"], df[group_col], df["Event_Observed"]
        )
        return {"test": "Log-rank multivarié", "p_value": result.p_value, "statistic": result.test_statistic}


def km_survival_table(kmf: KaplanMeierFitter, time_points: list) -> pd.DataFrame:
    rows = []
    for t in time_points:
        prob = kmf.predict(t)
        rows.append({"Temps (mois)": t, "P(survie)": f"{prob:.4f}", "P(survie) %": f"{prob*100:.2f}%"})
    return pd.DataFrame(rows)


# ── Nelson-Aalen ─────────────────────────────────────────────────────────────

def plot_na_global(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    naf = NelsonAalenFitter()
    naf.fit(df["Time_to_Event"], event_observed=df["Event_Observed"], label="Population globale")
    naf.plot_cumulative_hazard(ax=ax, ci_show=True, color=PALETTE[0])

    ax.set_title("Fonction de risque cumulée (Nelson-Aalen)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Temps (mois)")
    ax.set_ylabel("Risque cumulé H(t)")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_na_stratified(df: pd.DataFrame, group_col: str):
    groups = df[group_col].dropna().unique()
    fig, ax = plt.subplots(figsize=(10, 5))

    naf = NelsonAalenFitter()
    for i, grp in enumerate(sorted(groups, key=str)):
        mask = df[group_col] == grp
        naf.fit(
            df.loc[mask, "Time_to_Event"],
            event_observed=df.loc[mask, "Event_Observed"],
            label=str(grp),
        )
        naf.plot_cumulative_hazard(ax=ax, ci_show=True, color=PALETTE[i % len(PALETTE)])

    ax.set_title(f"Risque cumulé par {group_col}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Temps (mois)")
    ax.set_ylabel("Risque cumulé H(t)")
    ax.legend(title=group_col)
    fig.tight_layout()
    return fig


# ── Cox ──────────────────────────────────────────────────────────────────────

def plot_hazard_ratios(cph):
    summary = cph.summary.copy()
    summary = summary.sort_values("exp(coef)")

    fig, ax = plt.subplots(figsize=(9, 5))
    y_pos = range(len(summary))

    colors = ["#d62728" if hr > 1 else "#2ca02c" for hr in summary["exp(coef)"]]

    ax.barh(y_pos, summary["exp(coef)"] - 1, left=1, color=colors, alpha=0.7, height=0.5)
    ax.errorbar(
        summary["exp(coef)"],
        y_pos,
        xerr=[
            summary["exp(coef)"] - summary["exp(coef) lower 95%"],
            summary["exp(coef) upper 95%"] - summary["exp(coef)"],
        ],
        fmt="none",
        color="black",
        capsize=4,
        linewidth=1.5,
    )
    ax.scatter(summary["exp(coef)"], y_pos, color="black", zorder=5, s=30)
    ax.axvline(x=1, color="black", linestyle="--", linewidth=1)

    var_labels = {
        "Age": "Âge",
        "Sex_Female": "Sexe (Femme)",
        "Smoker": "Fumeur",
        "Treatment_Experimental": "Traitement Expérimental",
        "Activity_High": "Activité Physique (Haute)",
        "Activity_Moderate": "Activité Physique (Modérée)",
    }
    labels = [var_labels.get(v, v) for v in summary.index]
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Hazard Ratio (HR)")
    ax.set_title("Hazard Ratios — Modèle de Cox", fontsize=14, fontweight="bold")

    protective = mpatches.Patch(color="#2ca02c", alpha=0.7, label="Protecteur (HR < 1)")
    risk = mpatches.Patch(color="#d62728", alpha=0.7, label="Facteur de risque (HR > 1)")
    ax.legend(handles=[protective, risk], loc="lower right")
    fig.tight_layout()
    return fig


def plot_cox_adjusted_survival(cph, covariate: str, values: list, df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    from utils.data_loader import prepare_cox_data

    cox_data = prepare_cox_data(df)
    mean_profile = cox_data.drop(columns=["Time_to_Event", "Event_Observed"]).mean()

    for i, val in enumerate(values):
        profile = mean_profile.copy()
        profile[covariate] = val
        cph.plot_partial_effects_on_outcome(
            covariates=covariate,
            values=[val],
            ax=ax,
            label=f"{covariate} = {val}",
            plot_baseline=False,
        )

    ax.set_title(f"Courbes de survie ajustées par {covariate}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Temps (mois)")
    ax.set_ylabel("Probabilité de survie S(t)")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    return fig
