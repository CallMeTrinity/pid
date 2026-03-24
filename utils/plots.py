import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import KaplanMeierFitter, NelsonAalenFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

PALETTE = sns.color_palette("husl", 8)


# ── Kaplan-Meier ─────────────────────────────────────────────────────────────

def plot_km_global(df, time_col, event_col):
    fig, ax = plt.subplots(figsize=(10, 5))
    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col], event_observed=df[event_col], label="Population globale")
    kmf.plot_survival_function(ax=ax, ci_show=True, color=PALETTE[0])

    median = kmf.median_survival_time_
    ax.axvline(x=median, color="red", ls="--", alpha=.7, label=f"Mediane = {median:.1f}")
    ax.axhline(y=.5, color="red", ls="--", alpha=.3)
    ax.set(title="Courbe de survie globale (Kaplan-Meier)",
           xlabel="Temps (mois)", ylabel="S(t)", ylim=(0, 1.05))
    ax.legend()
    fig.tight_layout()
    return fig, kmf


def plot_km_stratified(df, time_col, event_col, group_col):
    groups = sorted(df[group_col].dropna().unique(), key=str)
    fig, ax = plt.subplots(figsize=(10, 5))
    kmf = KaplanMeierFitter()
    for i, g in enumerate(groups):
        m = df[group_col] == g
        kmf.fit(df.loc[m, time_col], event_observed=df.loc[m, event_col], label=str(g))
        kmf.plot_survival_function(ax=ax, ci_show=True, color=PALETTE[i % len(PALETTE)])
    ax.set(title=f"Survie par {group_col}", xlabel="Temps (mois)", ylabel="S(t)", ylim=(0, 1.05))
    ax.legend(title=group_col)
    fig.tight_layout()
    return fig


def km_survival_table(df, time_col, event_col, points=None):
    if points is None:
        points = [12, 24, 36, 60, 100]
    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col], event_observed=df[event_col])
    rows = []
    for t in points:
        p = float(kmf.predict(t))
        rows.append({"Temps (mois)": t, "S(t)": f"{p:.4f}", "%": f"{p*100:.2f}%"})
    return pd.DataFrame(rows), kmf


def km_median_by_group(df, time_col, event_col, group_col):
    groups = sorted(df[group_col].dropna().unique(), key=str)
    kmf = KaplanMeierFitter()
    rows = []
    for g in groups:
        m = df[group_col] == g
        kmf.fit(df.loc[m, time_col], event_observed=df.loc[m, event_col])
        rows.append({
            "Groupe": str(g),
            "n": int(m.sum()),
            "Evenements": int(df.loc[m, event_col].sum()),
            "Mediane (mois)": f"{kmf.median_survival_time_:.2f}",
        })
    return pd.DataFrame(rows)


def logrank_result(df, time_col, event_col, group_col):
    groups = sorted(df[group_col].dropna().unique(), key=str)
    if len(groups) == 2:
        m0 = df[group_col] == groups[0]
        m1 = df[group_col] == groups[1]
        r = logrank_test(
            df.loc[m0, time_col], df.loc[m1, time_col],
            event_observed_A=df.loc[m0, event_col],
            event_observed_B=df.loc[m1, event_col],
        )
        return {"test": "Log-rank", "stat": r.test_statistic, "p": r.p_value}
    else:
        r = multivariate_logrank_test(df[time_col], df[group_col], df[event_col])
        return {"test": "Log-rank multivarie", "stat": r.test_statistic, "p": r.p_value}


# ── Nelson-Aalen ─────────────────────────────────────────────────────────────

def plot_na_global(df, time_col, event_col):
    fig, ax = plt.subplots(figsize=(10, 5))
    naf = NelsonAalenFitter()
    naf.fit(df[time_col], event_observed=df[event_col], label="Population globale")
    naf.plot_cumulative_hazard(ax=ax, ci_show=True, color=PALETTE[0])
    ax.set(title="Risque cumule (Nelson-Aalen)", xlabel="Temps (mois)", ylabel="H(t)")
    ax.legend()
    fig.tight_layout()
    return fig, naf


def plot_na_stratified(df, time_col, event_col, group_col):
    groups = sorted(df[group_col].dropna().unique(), key=str)
    fig, ax = plt.subplots(figsize=(10, 5))
    naf = NelsonAalenFitter()
    for i, g in enumerate(groups):
        m = df[group_col] == g
        naf.fit(df.loc[m, time_col], event_observed=df.loc[m, event_col], label=str(g))
        naf.plot_cumulative_hazard(ax=ax, ci_show=True, color=PALETTE[i % len(PALETTE)])
    ax.set(title=f"Risque cumule par {group_col}", xlabel="Temps (mois)", ylabel="H(t)")
    ax.legend(title=group_col)
    fig.tight_layout()
    return fig


# ── Cox ──────────────────────────────────────────────────────────────────────

def plot_hazard_ratios(cph):
    summary = cph.summary.copy().sort_values("exp(coef)")
    fig, ax = plt.subplots(figsize=(9, 5))
    y = range(len(summary))
    colors = ["#d62728" if hr > 1 else "#2ca02c" for hr in summary["exp(coef)"]]

    ax.barh(y, summary["exp(coef)"] - 1, left=1, color=colors, alpha=.7, height=.5)
    ax.errorbar(
        summary["exp(coef)"], y,
        xerr=[
            summary["exp(coef)"] - summary["exp(coef) lower 95%"],
            summary["exp(coef) upper 95%"] - summary["exp(coef)"],
        ],
        fmt="none", color="black", capsize=4, lw=1.5,
    )
    ax.scatter(summary["exp(coef)"], y, color="black", zorder=5, s=30)
    ax.axvline(x=1, color="black", ls="--", lw=1)

    labels_map = {
        "Age": "Age", "Sex_Female": "Sexe (Femme)", "Smoker": "Fumeur",
        "Treatment_Experimental": "Trait. Experimental",
        "Activity_High": "Activite Haute", "Activity_Moderate": "Activite Moderee",
    }
    ax.set_yticks(list(y))
    ax.set_yticklabels([labels_map.get(v, v) for v in summary.index])
    ax.set_xlabel("Hazard Ratio")
    ax.set_title("Hazard Ratios — Modele de Cox", fontweight="bold")

    p1 = mpatches.Patch(color="#2ca02c", alpha=.7, label="Protecteur (HR<1)")
    p2 = mpatches.Patch(color="#d62728", alpha=.7, label="Risque (HR>1)")
    ax.legend(handles=[p1, p2], loc="lower right")
    fig.tight_layout()
    return fig
