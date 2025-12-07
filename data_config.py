import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
from lifelines import KaplanMeierFitter

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "METABRIC_RNA_Mutation.csv"

df = pd.read_csv(DATA_PATH, low_memory=False)


# -------------------------------------------------
# COLUMN GROUPS
# -------------------------------------------------
def map_cancer_event(value):
    """
    1 = death from breast cancer (cancer-specific event)
    0 = alive, lost to follow-up, or died of other causes
    """
    v = str(value).strip().lower()

    if v in ("", "nan", "none"):
        return 0

    # numeric / simple flags
    if v in {"1", "yes", "y", "true"}:
        return 1
    if v in {"0", "no", "n", "false"}:
        return 0

    # Textual descriptions from METABRIC-like sources
    if "died" in v and "disease" in v:
        return 1  # Died of Disease -> cancer-specific
    if "died" in v:
        return 0  # Died of other causes
    if "living" in v or "alive" in v:
        return 0

    return 0


if "death_from_cancer" in df.columns:
    df["death_from_cancer_event"] = df["death_from_cancer"].apply(map_cancer_event).astype(int)
else:
    df["death_from_cancer_event"] = np.nan

clinical_cols = [
    "patient_id",
    "age_at_diagnosis",
    "type_of_breast_surgery",
    "cancer_type",
    "cancer_type_detailed",
    "cellularity",
    "chemotherapy",
    "pam50_+_claudin-low_subtype",
    "cohort",
    "er_status_measured_by_ihc",
    "er_status",
    "neoplasm_histologic_grade",
    "her2_status_measured_by_snp6",
    "her2_status",
    "tumor_other_histologic_subtype",
    "hormone_therapy",
    "inferred_menopausal_state",
    "integrative_cluster",
    "primary_tumor_laterality",
    "lymph_nodes_examined_positive",
    "mutation_count",
    "nottingham_prognostic_index",
    "oncotree_code",
    "overall_survival_months",
    "overall_survival",
    "pr_status",
    "radio_therapy",
    "3-gene_classifier_subtype",
    "tumor_size",
    "tumor_stage",
    "death_from_cancer",
]
clinical_cols = [c for c in clinical_cols if c in df.columns]

# mutation columns (binary-ish)
mutation_cols = [
    c for c in df.columns
    if ("mut" in c.lower() or "mutation" in c.lower()) and c not in clinical_cols
]

# mRNA columns (z-scores)
mrna_cols = [
    c for c in df.columns
    if c not in clinical_cols and c not in mutation_cols
]

SENTINEL_ZERO = {"0", "0.0", "", "nan", "NaN", "None", "NONE", "null", "NULL"}


# -------------------------------------------------
# KM HELPERS
# -------------------------------------------------
def _km_axis_tuning(fig, durations):
    """Set nicer x/y ranges for KM curves."""
    max_time = np.nanpercentile(durations.values, 99.5)
    if not np.isfinite(max_time) or max_time <= 0:
        max_time = float(np.nanmax(durations.values))
    if not np.isfinite(max_time) or max_time <= 0:
        return fig

    fig.update_xaxes(
        range=[0, max_time * 1.02],
        title_text="Time since diagnosis (months)",
        showgrid=True,
        zeroline=False,
    )
    fig.update_yaxes(
        range=[0, 1.0],
        title_text="Cancer-specific survival probability",
        showgrid=True,
        zeroline=False,
    )
    return fig


def _build_event_overall_survival(df_sub):
    """Fallback event from overall_survival column."""
    vals = df_sub["overall_survival"].astype(str).str.lower().str.strip()
    dead_keys = ["dead", "died", "deceased", "1", "true", "yes"]
    return vals.apply(lambda v: any(k in v for k in dead_keys)).astype(int)


def make_km_figure(df_all: pd.DataFrame) -> go.Figure:
    """Overview KM curve: prefers cancer-specific, falls back to overall."""
    if "overall_survival_months" not in df_all.columns:
        return go.Figure().update_layout(title="overall_survival_months not found")

    durations = df_all["overall_survival_months"]
    event_observed = None
    title_prefix = "Survival"

    # Prefer cancer-specific death
    if "death_from_cancer_event" in df_all.columns and df_all["death_from_cancer_event"].notna().any():
        event_observed = df_all["death_from_cancer_event"]
        title_prefix = "Cancer-specific survival"
    elif "overall_survival" in df_all.columns:
        event_observed = _build_event_overall_survival(df_all)
        title_prefix = "Overall survival"

    if event_observed is None:
        return go.Figure().update_layout(title="No event column to build survival")

    if title_prefix.startswith("Cancer-specific"):
        subtitle = (
            "Shows the probability that a patient has not died from breast cancer at each time point."
        )
    else:
        subtitle = (
            "Shows the probability that a patient is still alive at each time point; "
            "any death is treated as an event."
        )

    kmf = KaplanMeierFitter()
    kmf.fit(durations, event_observed=event_observed, label="KM estimate")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=kmf.survival_function_.index,
            y=kmf.survival_function_["KM estimate"],
            mode="lines",
            line=dict(width=3),
            name="KM estimate",
        )
    )

    if kmf.confidence_interval_ is not None:
        ci = kmf.confidence_interval_
        fig.add_trace(go.Scatter(
            x=ci.index,
            y=ci.iloc[:, 0],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=ci.index,
            y=ci.iloc[:, 1],
            mode="lines",
            fill="tonexty",
            line=dict(width=0),
            name="95% CI",
            opacity=0.2,
        ))

    fig.update_layout(
        title=f"{title_prefix} – Kaplan–Meier curve<br><sup>{subtitle}</sup>",
        template="plotly_white",
        height=360,
        margin=dict(l=60, r=20, t=60, b=60),
    )
    fig = _km_axis_tuning(fig, durations)
    return fig


def make_km_for_subset(sub_df, mut_mask, title_suffix=""):
    """
    KM curves for mutated vs wild-type within a subset.

    Uses:
      - overall_survival_months as duration
      - death_from_cancer_event as cancer-specific event when available,
        otherwise overall_survival (alive/dead).
    """
    if "overall_survival_months" not in sub_df.columns:
        return go.Figure().update_layout(title="overall_survival_months not found")

    durations = sub_df["overall_survival_months"]
    event = None
    title_prefix = "Survival"

    if "death_from_cancer_event" in sub_df.columns and sub_df["death_from_cancer_event"].notna().any():
        event = sub_df["death_from_cancer_event"]
        title_prefix = "Cancer-specific survival"
    elif "overall_survival" in sub_df.columns:
        event = _build_event_overall_survival(sub_df)
        title_prefix = "Overall survival"

    if event is None:
        return go.Figure().update_layout(title="No event column to build survival")

    fig = go.Figure()
    kmf = KaplanMeierFitter()

    # mutated
    if mut_mask.sum() > 0:
        kmf.fit(durations[mut_mask], event_observed=event[mut_mask], label="Mutated")
        fig.add_trace(go.Scatter(
            x=kmf.survival_function_.index,
            y=kmf.survival_function_["Mutated"],
            mode="lines",
            name="Mutated",
            line=dict(width=3)
        ))
    # wild-type
    if (~mut_mask).sum() > 0:
        kmf.fit(durations[~mut_mask], event_observed=event[~mut_mask], label="Wild-type")
        fig.add_trace(go.Scatter(
            x=kmf.survival_function_.index,
            y=kmf.survival_function_["Wild-type"],
            mode="lines",
            name="Wild-type",
            line=dict(width=3, dash="dash")
        ))

    fig.update_layout(
        title=f"{title_prefix} by mutation status {title_suffix}",
        template="plotly_white",
        height=320,
        margin=dict(l=60, r=20, t=60, b=40),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.7)"),
    )
    fig = _km_axis_tuning(fig, durations)
    return fig
