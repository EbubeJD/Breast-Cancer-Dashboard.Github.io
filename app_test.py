import pandas as pd
import numpy as np
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go

from lifelines import KaplanMeierFitter
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

df = pd.read_csv(r"data/METABRIC_RNA_Mutation.csv", low_memory=False)

# Normalize column names a bit
df.columns = [c.strip() for c in df.columns]

# A few convenience derived columns
if "age_at_diagnosis" in df.columns:
    df["age_at_diagnosis"] = pd.to_numeric(df["age_at_diagnosis"], errors="coerce")

if "overall_survival_months" in df.columns:
    df["overall_survival_months"] = pd.to_numeric(
        df["overall_survival_months"], errors="coerce"
    )

# A rough guess at which columns are clinical vs genetic.
# This can be tuned, but works OK for METABRIC.
clinical_cols = [
    c
    for c in df.columns
    if any(
        key in c.lower()
        for key in [
            "age",
            "tumor",
            "stage",
            "grade",
            "lymph",
            "node",
            "er_status",
            "pr_status",
            "her2",
            "os",
            "overall_survival",
            "death",
            "surgery",
            "npi",
            "breast_cancer",
        ]
    )
]

# mutation columns (binary-ish)
mutation_cols = [
    c
    for c in df.columns
    if ("mut" in c.lower() or "mutation" in c.lower()) and c not in clinical_cols
]

# mRNA columns (z-scores)
mrna_cols = [c for c in df.columns if c not in clinical_cols and c not in mutation_cols]

SENTINEL_ZERO = {"0", "0.0", "", "nan", "NaN", "None", "NONE", "null", "NULL"}

# -------------------------------------------------
# PRECOMPUTED MATRICES FOR PERFORMANCE
# -------------------------------------------------
# Precompute mRNA numeric matrix, variance, correlations, and PCA
MRNA_MATRIX = None
MRNA_VAR = None
MRNA_CORR_ALL = None
MRNA_MAX_PCS = 0
MRNA_PCA = None
MRNA_PCA_FULL = None

if mrna_cols:
    _mrna = df[mrna_cols].apply(pd.to_numeric, errors="coerce")
    _mrna = _mrna.dropna(axis=1, how="all")
    _mrna = _mrna.fillna(_mrna.median())
    MRNA_MATRIX = _mrna
    if not _mrna.empty:
        MRNA_VAR = _mrna.var().sort_values(ascending=False)
        MRNA_CORR_ALL = _mrna.corr()
        MRNA_MAX_PCS = min(10, _mrna.shape[1])
        _scaler = StandardScaler()
        _mrna_scaled = _scaler.fit_transform(_mrna.values)
        MRNA_PCA = PCA(n_components=MRNA_MAX_PCS, random_state=0).fit(_mrna_scaled)
        MRNA_PCA_FULL = MRNA_PCA.transform(_mrna_scaled)

# Precompute mutation binary matrix, counts, and correlations
MUT_BIN = None
MUT_COUNTS = None
MUT_CORR_ALL = None

if mutation_cols:
    _mut_str = df[mutation_cols].astype(str).apply(lambda s: s.str.strip())
    MUT_BIN = (~_mut_str.isin(SENTINEL_ZERO)).astype(int)
    if not MUT_BIN.empty:
        MUT_COUNTS = MUT_BIN.sum().sort_values(ascending=False)
        MUT_CORR_ALL = MUT_BIN.corr()

# -------------------------------------------------
# KM HELPERS
# -------------------------------------------------


def _km_axis_tuning(fig, durations):
    """Nice x-axis (months) based on range of durations."""
    dur = pd.to_numeric(durations, errors="coerce").dropna()
    if len(dur) == 0:
        return fig
    max_m = float(dur.max())
    if max_m <= 24:
        step = 3
    elif max_m <= 60:
        step = 6
    else:
        step = 12
    fig.update_xaxes(
        title="Months from diagnosis",
        dtick=step,
        showgrid=True,
        zeroline=False,
    )
    fig.update_yaxes(
        title="Survival probability",
        range=[0, 1.05],
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
    """Overview KM curve: prefers cancer-specific, falls back
    to overall."""
    if "overall_survival_months" not in df_all.columns:
        return go.Figure().update_layout(
            title="overall_survival_months not found"
        )

    durations = df_all["overall_survival_months"]
    event_observed = None
    title_prefix = "Survival"

    # Prefer cancer-specific death
    if "death_from_cancer_event" in df_all.columns and df_all[
        "death_from_cancer_event"
    ].notna().any():
        event_observed = df_all["death_from_cancer_event"]
        title_prefix = "Cancer-specific survival"
    elif "overall_survival" in df_all.columns:
        event_observed = _build_event_overall_survival(df_all)
        title_prefix = "Overall survival"

    if event_observed is None:
        return go.Figure().update_layout(
            title="No suitable event column found for KM curve."
        )

    kmf = KaplanMeierFitter()
    dur = pd.to_numeric(durations, errors="coerce")
    mask = dur.notna() & event_observed.notna()
    dur = dur[mask]
    ev = event_observed[mask]

    if len(dur) < 5:
        return go.Figure().update_layout(
            title="Not enough data to build survival curve."
        )

    kmf.fit(dur, event_observed=ev, label="All patients")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=kmf.survival_function_.index,
            y=kmf.survival_function_["All patients"],
            mode="lines",
            name="All patients",
            line=dict(width=3),
        )
    )

    fig.update_layout(
        title=f"{title_prefix} for METABRIC cohort",
        template="plotly_white",
        height=320,
        margin=dict(l=60, r=20, t=60, b=40),
        legend=dict(
            x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.7)"
        ),
    )
    fig = _km_axis_tuning(fig, dur)
    return fig


def make_km_by_mutation(df_all: pd.DataFrame, gene_mut_col: str) -> go.Figure:
    """KM by mutated vs wild-type for a given binary mutation column."""
    col = gene_mut_col
    if col not in df_all.columns:
        return go.Figure().update_layout(
            title=f"Column {col} not found for survival by mutation."
        )

    if "overall_survival_months" not in df_all.columns:
        return go.Figure().update_layout(
            title="overall_survival_months not found."
        )

    durations = pd.to_numeric(
        df_all["overall_survival_months"], errors="coerce"
    )
    mut_mask_raw = df_all[col]
    # robust: interpret non-zero / non-empty as mutated
    mut_mask = (
        (~mut_mask_raw.astype(str).str.strip().isin(SENTINEL_ZERO))
        & mut_mask_raw.notna()
    )

    if "death_from_cancer_event" in df_all.columns:
        event = df_all["death_from_cancer_event"]
        title_prefix = "Cancer-specific survival"
    elif "overall_survival" in df_all.columns:
        event = _build_event_overall_survival(df_all)
        title_prefix = "Overall survival"
    else:
        return go.Figure().update_layout(
            title="No suitable event column for survival by mutation."
        )

    mask = durations.notna() & event.notna()
    durations = durations[mask]
    event = event[mask]
    mut_mask = mut_mask[mask]

    if len(durations) < 10:
        return go.Figure().update_layout(
            title="Not enough data for survival by mutation."
        )

    kmf = KaplanMeierFitter()
    fig = go.Figure()

    # mutated
    if mut_mask.sum() > 0:
        kmf.fit(
            durations[mut_mask],
            event_observed=event[mut_mask],
            label="Mutated",
        )
        fig.add_trace(
            go.Scatter(
                x=kmf.survival_function_.index,
                y=kmf.survival_function_["Mutated"],
                mode="lines",
                name="Mutated",
                line=dict(width=3),
            )
        )
    # wild-type
    if (~mut_mask).sum() > 0:
        kmf.fit(
            durations[~mut_mask],
            event_observed=event[~mut_mask],
            label="Wild-type",
        )
        fig.add_trace(
            go.Scatter(
                x=kmf.survival_function_.index,
                y=kmf.survival_function_["Wild-type"],
                mode="lines",
                name="Wild-type",
                line=dict(width=3, dash="dash"),
            )
        )

    fig.update_layout(
        title=f"{title_prefix} by mutation status ({gene_mut_col})",
        template="plotly_white",
        height=320,
        margin=dict(l=60, r=20, t=60, b=40),
        legend=dict(
            x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.7)"
        ),
    )
    fig = _km_axis_tuning(fig, durations)
    return fig

# -------------------------------------------------
# APP LAYOUT
# -------------------------------------------------

app = Dash(__name__)
server = app.server

# Overview charts, mutation tab, mRNA tab, co-occurrence tab
app.layout = html.Div(
    [
        html.H1("METABRIC Breast Cancer Genomic Dashboard"),
        dcc.Tabs(
            id="tab-selector",
            value="overview",
            children=[
                dcc.Tab(
                    label="Overview",
                    value="overview",
                    children=[
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H3("Clinical overview"),
                                        dcc.Graph(id="age-hist"),
                                        dcc.Graph(id="stage-bar"),
                                        dcc.Graph(id="surgery-pie"),
                                        dcc.Graph(id="npi-hist"),
                                    ],
                                    style={"width": "50%", "display": "inline-block"},
                                ),
                                html.Div(
                                    [
                                        html.H3("Genomic overview"),
                                        dcc.Graph(id="top-mut-bar"),
                                        dcc.Graph(id="mut-heatmap"),
                                        dcc.Graph(id="overview-km"),
                                    ],
                                    style={"width": "50%", "display": "inline-block"},
                                ),
                            ]
                        )
                    ],
                ),
                dcc.Tab(
                    label="Mutations",
                    value="mutation",
                    children=[
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label("Select mutation gene"),
                                        dcc.Dropdown(
                                            id="mut-gene-dropdown",
                                            options=[
                                                {"label": c, "value": c}
                                                for c in sorted(mutation_cols)
                                            ],
                                            value=mutation_cols[0]
                                            if mutation_cols
                                            else None,
                                            placeholder="Select a mutation gene",
                                            clearable=False,
                                        ),
                                        html.Div(
                                            id="mut-summary",
                                            style={
                                                "marginTop": "10px",
                                                "marginBottom": "20px",
                                            },
                                        ),
                                        dcc.Graph(id="mut-pie"),
                                        dcc.Graph(id="mut-by-pam50"),
                                        dcc.Graph(id="mut-by-er"),
                                        dcc.Graph(id="mut-treatment"),
                                        dcc.Graph(id="mut-burden"),
                                        dcc.Graph(id="mut-survival"),
                                    ],
                                    style={"width": "60%", "display": "inline-block"},
                                ),
                                html.Div(
                                    [
                                        html.H4(
                                            "Notes on mutation tab", style={"marginTop": "0"}
                                        ),
                                        html.P(
                                            """
                                            This tab focuses on binary mutation status (0/1) for a chosen gene.
                                            We look at its frequency overall and by subtype / ER status, its
                                            association with treatment, and survival differences.
                                            """
                                        ),
                                    ],
                                    style={
                                        "width": "38%",
                                        "display": "inline-block",
                                        "verticalAlign": "top",
                                        "marginLeft": "2%",
                                    },
                                ),
                            ]
                        )
                    ],
                ),
                dcc.Tab(
                    label="mRNA Expression",
                    value="mrna",
                    children=[
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label("Select mRNA gene"),
                                        dcc.Dropdown(
                                            id="mrna-gene-dropdown",
                                            options=[
                                                {"label": c, "value": c}
                                                for c in sorted(mrna_cols)
                                            ],
                                            value=mrna_cols[0]
                                            if mrna_cols
                                            else None,
                                            placeholder="Select an mRNA gene",
                                            clearable=False,
                                        ),
                                        html.Label(
                                            "Number of PCA components (2-10)",
                                            style={"marginTop": "10px"},
                                        ),
                                        dcc.Slider(
                                            id="mrna-pca-components",
                                            min=2,
                                            max=10,
                                            step=1,
                                            value=5,
                                            marks={i: str(i) for i in range(2, 11)},
                                        ),
                                        html.Label(
                                            "Number of K-means clusters (k)",
                                            style={"marginTop": "10px"},
                                        ),
                                        dcc.Slider(
                                            id="mrna-k",
                                            min=2,
                                            max=8,
                                            step=1,
                                            value=3,
                                            marks={i: str(i) for i in range(2, 9)},
                                        ),
                                    ],
                                    style={"marginBottom": "20px"},
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [dcc.Graph(id="mrna-gene-box")],
                                            style={"width": "50%", "display": "inline-block"},
                                        ),
                                        html.Div(
                                            [dcc.Graph(id="mrna-corr-bar")],
                                            style={"width": "50%", "display": "inline-block"},
                                        ),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [dcc.Graph(id="mrna-pca-scatter")],
                                            style={"width": "60%", "display": "inline-block"},
                                        ),
                                        html.Div(
                                            [dcc.Graph(id="mrna-pca-var")],
                                            style={"width": "38%", "display": "inline-block"},
                                        ),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [dcc.Graph(id="mrna-kmeans-scatter")],
                                            style={"width": "60%", "display": "inline-block"},
                                        ),
                                        html.Div(
                                            [html.Div(id="mrna-kmeans-summary")],
                                            style={"width": "38%", "display": "inline-block"},
                                        ),
                                    ]
                                ),
                            ]
                        )
                    ],
                ),
                dcc.Tab(
                    label="Co-occurrence / Co-expression",
                    value="co",
                    children=[
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Label(
                                            "Top N mutated genes for co-mutation analysis"
                                        ),
                                        dcc.Slider(
                                            id="co-top-n",
                                            min=5,
                                            max=30,
                                            step=1,
                                            value=15,
                                            marks={
                                                5: "5",
                                                10: "10",
                                                15: "15",
                                                20: "20",
                                                25: "25",
                                                30: "30",
                                            },
                                        ),
                                        html.Label(
                                            "Top N variable mRNA genes for co-expression",
                                            style={"marginTop": "10px"},
                                        ),
                                        dcc.Slider(
                                            id="co-mrna-topn",
                                            min=5,
                                            max=50,
                                            step=1,
                                            value=20,
                                            marks={
                                                5: "5",
                                                10: "10",
                                                20: "20",
                                                30: "30",
                                                40: "40",
                                                50: "50",
                                            },
                                        ),
                                    ],
                                    style={"marginBottom": "20px"},
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [dcc.Graph(id="co-heatmap")],
                                            style={"width": "48%", "display": "inline-block"},
                                        ),
                                        html.Div(
                                            [dcc.Graph(id="co-mut-network")],
                                            style={
                                                "width": "48%",
                                                "display": "inline-block",
                                                "marginLeft": "2%",
                                            },
                                        ),
                                    ]
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [dcc.Graph(id="co-mrna-heatmap")],
                                            style={"width": "48%", "display": "inline-block"},
                                        ),
                                        html.Div(
                                            [dcc.Graph(id="co-network")],
                                            style={
                                                "width": "48%",
                                                "display": "inline-block",
                                                "marginLeft": "2%",
                                            },
                                        ),
                                    ]
                                ),
                            ]
                        )
                    ],
                ),
            ],
        ),
    ]
)

# -------------------------------------------------
# OVERVIEW CALLBACK
# -------------------------------------------------


@app.callback(
    Output("age-hist", "figure"),
    Output("stage-bar", "figure"),
    Output("surgery-pie", "figure"),
    Output("npi-hist", "figure"),
    Output("top-mut-bar", "figure"),
    Output("mut-heatmap", "figure"),
    Output("overview-km", "figure"),
    Input("tab-selector", "value"),
)
def update_overview(tab):
    if tab != "overview":
        # Return empty figs (Dash requires something)
        empty = go.Figure().update_layout(
            template="plotly_white", height=300
        )
        return empty, empty, empty, empty, empty, empty, empty

    # Age histogram
    if "age_at_diagnosis" in df.columns:
        age_data = pd.to_numeric(
            df["age_at_diagnosis"], errors="coerce"
        ).dropna()
        age_fig = px.histogram(
            age_data,
            nbins=20,
            title="Age at diagnosis",
        )
        age_fig.update_layout(
            xaxis_title="Age (years)",
            yaxis_title="Number of patients",
            template="plotly_white",
            height=340,
            margin=dict(l=40, r=10, t=60, b=60),
        )
    else:
        age_fig = None

    # Tumor stage 0-4
    if "tumor_stage" in df.columns:
        stage_df = df.copy()

        def _stage_label(v):
            if pd.isna(v):
                return "Unknown"
            s = str(v).strip().upper()
            for candidate in ["0", "I", "II", "III", "IV"]:
                if candidate in s:
                    return candidate
            return "Other"

        stage_df["tumor_stage_bucket"] = stage_df["tumor_stage"].apply(
            _stage_label
        )
        stage_counts = (
            stage_df["tumor_stage_bucket"]
            .value_counts()
            .reindex(["0", "I", "II", "III", "IV", "Other", "Unknown"], fill_value=0)
        )
        stage_fig = px.bar(
            x=stage_counts.index,
            y=stage_counts.values,
            title="Tumor stage distribution (rough buckets)",
        )
        stage_fig.update_layout(
            xaxis_title="Stage",
            yaxis_title="Number of patients",
            template="plotly_white",
            height=340,
            margin=dict(l=40, r=10, t=60, b=60),
        )
    else:
        stage_fig = None

    # Surgery type
    if "type_of_breast_surgery" in df.columns:
        surgery_fig = px.pie(
            df,
            names="type_of_breast_surgery",
            title="Type of Breast Surgery",
            hole=0.4,
        )
        surgery_fig.update_traces(textinfo="percent+label")
        surgery_fig.update_layout(
            legend_title_text="Surgery type",
            template="plotly_white",
            height=340,
            margin=dict(l=20, r=20, t=60, b=20),
        )
    else:
        surgery_fig = None

    # Nottingham Prognostic Index
    if "nottingham_prognostic_index" in df.columns:
        npi = pd.to_numeric(
            df["nottingham_prognostic_index"], errors="coerce"
        ).dropna()
        npi_fig = px.histogram(
            npi,
            nbins=20,
            title="Nottingham Prognostic Index (NPI)",
        )
        npi_fig.update_layout(
            xaxis_title="NPI score",
            yaxis_title="Number of patients",
            template="plotly_white",
            height=340,
            margin=dict(l=40, r=10, t=60, b=60),
        )
    else:
        npi_fig = None

    # Top mutations + heatmap
    mut_fig = None
    heat_fig = None
    if mutation_cols and MUT_BIN is not None and MUT_COUNTS is not None:
        mutation_counts_clean = MUT_COUNTS
        top10_cols = mutation_counts_clean.head(10).index.tolist()
        top10 = mutation_counts_clean.head(10).reset_index()
        top10.columns = ["Gene", "Number of patients with mutation"]

        mut_fig = px.bar(
            top10,
            x="Gene",
            y="Number of patients with mutation",
            title="Top 10 mutated genes",
        )
        mut_fig.update_layout(
            xaxis_title="Mutation column",
            yaxis_title="Count",
            template="plotly_white",
            height=340,
            margin=dict(l=40, r=10, t=60, b=60),
        )

        # Heatmap of co-mutation for these columns
        corr_top10 = MUT_CORR_ALL.loc[top10_cols, top10_cols]
        heat_fig = px.imshow(
            corr_top10,
            x=[c.replace("_mut", "").upper() for c in corr_top10.columns],
            y=[c.replace("_mut", "").upper() for c in corr_top10.index],
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
            title="Co-mutation correlation (top 10)",
        )
        heat_fig.update_layout(
            template="plotly_white",
            height=340,
            margin=dict(l=80, r=10, t=60, b=60),
        )
        heat_fig.update_xaxes(tickangle=-45)

    overview_km = make_km_figure(df)

    return (
        age_fig,
        stage_fig,
        surgery_fig,
        npi_fig,
        mut_fig,
        heat_fig,
        overview_km,
    )


# -------------------------------------------------
# MRNA TAB CALLBACK (OPTIMIZED)
# -------------------------------------------------


@app.callback(
    Output("mrna-gene-box", "figure"),
    Output("mrna-corr-bar", "figure"),
    Output("mrna-pca-scatter", "figure"),
    Output("mrna-pca-var", "figure"),
    Output("mrna-kmeans-scatter", "figure"),
    Output("mrna-kmeans-summary", "children"),
    Input("mrna-gene-dropdown", "value"),
    Input("mrna-pca-components", "value"),
    Input("mrna-k", "value"),
)
def update_mrna_tab(gene, n_components, k):
    def empty_fig(title):
        fig = go.Figure()
        fig.update_layout(title=title, template="plotly_white", height=340)
        return fig

    if not mrna_cols or gene is None or gene not in df.columns:
        msg = "Select an mRNA gene to begin."
        return (
            empty_fig(msg),
            empty_fig(msg),
            empty_fig(msg),
            empty_fig(msg),
            empty_fig(msg),
            "Select an mRNA gene, PCA components, and k for clustering.",
        )

    # Expression view
    if "pam50_+_claudin-low_subtype" in df.columns:
        expr_fig = px.box(
            df,
            x="pam50_+_claudin-low_subtype",
            y=gene,
            title=f"Expression of {gene} by PAM50 subtype",
        )
        expr_fig.update_layout(
            xaxis_title="PAM50 subtype",
            yaxis_title=f"{gene} mRNA z-score",
            template="plotly_white",
            height=340,
            margin=dict(l=60, r=20, t=60, b=60),
        )
    else:
        expr_fig = px.box(
            df,
            y=gene,
            title=f"Expression of {gene}",
        )
        expr_fig.update_layout(
            yaxis_title=f"{gene} mRNA z-score",
            template="plotly_white",
            height=340,
            margin=dict(l=60, r=20, t=60, b=60),
        )

    # Numeric mRNA matrix (precomputed for performance)
    if MRNA_MATRIX is None:
        X = df[mrna_cols].apply(pd.to_numeric, errors="coerce")
        X = X.dropna(axis=1, how="all")
        X = X.fillna(X.median())
    else:
        X = MRNA_MATRIX

    if gene not in X.columns:
        corr_fig = empty_fig("Selected gene is not numeric in mRNA matrix.")
    else:
        corrs = X.corrwith(X[gene]).dropna().sort_values(ascending=False)
        corrs = corrs.drop(gene, errors="ignore")
        top_corr = corrs.head(15).sort_values()

        corr_df = top_corr.reset_index()
        corr_df.columns = ["Gene", "Correlation with " + gene]

        corr_fig = px.bar(
            corr_df,
            x="Correlation with " + gene,
            y="Gene",
            orientation="h",
            title=f"Top correlated genes with {gene}",
        )
        corr_fig.update_layout(
            xaxis_title="Pearson correlation",
            yaxis_title="Gene",
            template="plotly_white",
            height=340,
            margin=dict(l=120, r=40, t=60, b=40),
        )

    # PCA + K-means (reuse precomputed PCA where possible)
    n_features = X.shape[1]
    if n_components is None:
        n_components = 5

    max_pcs_allowed = MRNA_MAX_PCS if MRNA_MAX_PCS else min(10, n_features)
    n_components = int(max(2, min(n_components, max_pcs_allowed)))

    if (
        MRNA_PCA_FULL is not None
        and MRNA_PCA is not None
        and MRNA_PCA_FULL.shape[1] >= n_components
    ):
        X_pca = MRNA_PCA_FULL[:, :n_components]
        explained = MRNA_PCA.explained_variance_ratio_[:n_components]
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=n_components, random_state=0)
        X_pca = pca.fit_transform(X_scaled)
        explained = pca.explained_variance_ratio_

    pca_df = pd.DataFrame(
        {
            "PC1": X_pca[:, 0],
            "PC2": X_pca[:, 1],
        }
    )
    if "pam50_+_claudin-low_subtype" in df.columns:
        pca_df["Subtype"] = df[
            "pam50_+_claudin-low_subtype"
        ].fillna("Unknown")
    else:
        pca_df["Subtype"] = "All"

    pc1_var = explained[0] * 100
    pc2_var = explained[1] * 100

    pca_scatter = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="Subtype",
        title=f"PCA of mRNA expression (PC1 vs PC2, {n_components} PCs fitted)",
        opacity=0.8,
    )
    pca_scatter.update_layout(
        xaxis_title=f"PC1 ({pc1_var:.1f}% variance)",
        yaxis_title=f"PC2 ({pc2_var:.1f}% variance)",
        template="plotly_white",
        height=340,
        margin=dict(l=60, r=20, t=60, b=60),
    )

    comp_labels = [f"PC{i+1}" for i in range(n_components)]
    var_fig = px.bar(
        x=comp_labels,
        y=explained,
        labels={"x": "Principal component", "y": "Explained variance ratio"},
        title="Explained variance by PCA component",
    )
    var_fig.update_layout(
        template="plotly_white",
        height=340,
        margin=dict(l=60, r=20, t=60, b=60),
    )

    # K-means clustering on PCA space
    if k is None:
        k = 3
    k = int(max(2, min(k, 8)))

    k_dim = min(5, n_components)
    if k_dim < 2:
        k_dim = 2
    X_k = X_pca[:, :k_dim]

    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto")
    labels = kmeans.fit_predict(X_k)
    pca_df["Cluster"] = labels.astype(str)

    try:
        sil = silhouette_score(X_k, labels)
        sil_text = f"Silhouette score (higher is better): {sil:.3f}"
    except Exception:
        sil_text = "Silhouette score could not be computed."

    kmeans_fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="Cluster",
        title=f"K-means clustering on mRNA PCA space (k={k}, first {k_dim} PCs)",
        opacity=0.8,
    )
    kmeans_fig.update_layout(
        xaxis_title=f"PC1 ({pc1_var:.1f}% variance)",
        yaxis_title=f"PC2 ({pc2_var:.1f}% variance)",
        template="plotly_white",
        height=340,
        margin=dict(l=60, r=20, t=60, b=60),
    )

    lines = []
    if "Subtype" in pca_df.columns:
        comp_table = (
            pca_df.groupby(["Cluster", "Subtype"])
            .size()
            .unstack(fill_value=0)
        )
        for cluster_label, row in comp_table.iterrows():
            pieces = [f"Cluster {cluster_label}:"]
            for subtype, count in row.items():
                pieces.append(f"{subtype}={count}")
            lines.append("  " + ", ".join(pieces))

    summary_text = html.Div(
        [
            html.Div(
                sil_text,
                style={"fontWeight": "bold", "marginBottom": "8px"},
            ),
            html.Div(
                "K-means clustering is performed on the PCA-transformed mRNA matrix."
            ),
            html.Ul(
                [
                    html.Li(
                        f"Number of PCA components used: {n_components}"
                    ),
                    html.Li(f"Number of clusters (k): {k}"),
                ]
            ),
            html.Div(
                "Cluster composition by subtype:",
                style={"marginTop": "8px", "fontWeight": "bold"},
            ),
            html.Pre(
                "\n".join(lines)
                if lines
                else "Subtype labels not available."
            ),
        ]
    )

    return (
        expr_fig,
        corr_fig,
        pca_scatter,
        var_fig,
        kmeans_fig,
        summary_text,
    )


# -------------------------------------------------
# MUTATION TAB CALLBACK
# -------------------------------------------------


@app.callback(
    Output("mut-summary", "children"),
    Output("mut-pie", "figure"),
    Output("mut-by-pam50", "figure"),
    Output("mut-by-er", "figure"),
    Output("mut-treatment", "figure"),
    Output("mut-burden", "figure"),
    Output("mut-survival", "figure"),
    Input("mut-gene-dropdown", "value"),
)
def update_mut_tab(gene_col):
    if (
        not mutation_cols
        or gene_col is None
        or gene_col not in mutation_cols
    ):
        msg = "Select a mutation gene to begin."
        empty = go.Figure().update_layout(
            title=msg, template="plotly_white", height=320
        )
        return msg, empty, empty, empty, empty, empty, empty

    col = gene_col
    raw = df[col].astype(str).str.strip()
    mut_flag = (
        ~raw.isin(SENTINEL_ZERO) & raw.notna()
    )  # mutated if not in sentinel
    n_mut = mut_flag.sum()
    n_total = len(mut_flag)
    freq = n_mut / n_total if n_total > 0 else 0.0

    summary_html = html.Div(
        [
            html.Div(
                f"Column: {col}",
                style={"fontWeight": "bold", "marginBottom": "4px"},
            ),
            html.Div(f"Patients with mutation: {n_mut} / {n_total}"),
            html.Div(f"Mutation frequency: {freq:.1%}"),
        ]
    )

    # Pie overall
    mut_df = pd.DataFrame(
        {"status": np.where(mut_flag, "Mutated", "Wild-type")}
    )
    mut_pie = px.pie(
        mut_df,
        names="status",
        title=f"Overall mutation status for {col}",
        hole=0.4,
    )
    mut_pie.update_traces(textinfo="percent+label")
    mut_pie.update_layout(
        template="plotly_white",
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
    )

    # PAM50 subtype
    if "pam50_+_claudin-low_subtype" in df.columns:
        pam = df["pam50_+_claudin-low_subtype"].fillna("Unknown")
        pam_mut = pd.crosstab(pam, mut_flag)
        pam_mut.columns = ["Wild-type", "Mutated"]
        pam_mut = pam_mut.reset_index().rename(
            columns={"pam50_+_claudin-low_subtype": "Subtype"}
        )
        pam_melt = pam_mut.melt(
            id_vars="Subtype", value_vars=["Wild-type", "Mutated"], var_name="Status", value_name="Count"
        )
        pam_fig = px.bar(
            pam_melt,
            x="Subtype",
            y="Count",
            color="Status",
            barmode="group",
            title=f"Mutation status of {col} by PAM50 subtype",
        )
        pam_fig.update_layout(
            template="plotly_white",
            height=320,
            margin=dict(l=40, r=10, t=60, b=60),
        )
    else:
        pam_fig = go.Figure().update_layout(
            title="PAM50 subtype column not found.",
            template="plotly_white",
            height=320,
        )

    # ER status
    er_fig = go.Figure().update_layout(
        title="ER status column not found.",
        template="plotly_white",
        height=320,
    )
    er_col_candidates = [
        c
        for c in df.columns
        if "er_status" in c.lower() or "er status" in c.lower()
    ]
    if er_col_candidates:
        er_col = er_col_candidates[0]
        er = df[er_col].fillna("Unknown")
        er_mut = pd.crosstab(er, mut_flag)
        er_mut.columns = ["Wild-type", "Mutated"]
        er_mut = er_mut.reset_index().rename(columns={er_col: "ER status"})
        er_melt = er_mut.melt(
            id_vars="ER status",
            value_vars=["Wild-type", "Mutated"],
            var_name="Status",
            value_name="Count",
        )
        er_fig = px.bar(
            er_melt,
            x="ER status",
            y="Count",
            color="Status",
            barmode="group",
            title=f"Mutation status of {col} by ER status",
        )
        er_fig.update_layout(
            template="plotly_white",
            height=320,
            margin=dict(l=40, r=10, t=60, b=60),
        )

    # Treatment
    treat_fig = go.Figure().update_layout(
        title="Treatment column not found.",
        template="plotly_white",
        height=320,
    )
    treat_cols = [
        c
        for c in df.columns
        if "chemo" in c.lower()
        or "hormone" in c.lower()
        or "trastuzumab" in c.lower()
        or "therapy" in c.lower()
    ]
    if treat_cols:
        tcol = treat_cols[0]
        treat = df[tcol].fillna("Unknown")
        treat_mut = pd.crosstab(treat, mut_flag)
        treat_mut.columns = ["Wild-type", "Mutated"]
        treat_mut = treat_mut.reset_index().rename(
            columns={tcol: "Treatment category"}
        )
        treat_melt = treat_mut.melt(
            id_vars="Treatment category",
            value_vars=["Wild-type", "Mutated"],
            var_name="Status",
            value_name="Count",
        )
        treat_fig = px.bar(
            treat_melt,
            x="Treatment category",
            y="Count",
            color="Status",
            barmode="group",
            title=f"Mutation status of {col} by treatment",
        )
        treat_fig.update_layout(
            template="plotly_white",
            height=320,
            margin=dict(l=40, r=10, t=60, b=80),
            xaxis_tickangle=-45,
        )

    # Mutation burden by subtype
    burden_fig = go.Figure().update_layout(
        title="Mutation burden distribution",
        template="plotly_white",
        height=320,
    )
    if mutation_cols:
        mut_str_all = df[mutation_cols].astype(str).apply(
            lambda s: s.str.strip()
        )
        mut_bin_all = (
            ~mut_str_all.isin(SENTINEL_ZERO)
        ).astype(int)
        df["mutation_burden"] = mut_bin_all.sum(axis=1)

        if "pam50_+_claudin-low_subtype" in df.columns:
            burden_fig = px.box(
                df,
                x="pam50_+_claudin-low_subtype",
                y="mutation_burden",
                title="Overall mutation burden by PAM50 subtype",
            )
            burden_fig.update_layout(
                xaxis_title="PAM50 subtype",
                yaxis_title="Number of mutated genes",
                template="plotly_white",
                height=320,
                margin=dict(l=40, r=10, t=60, b=60),
            )
        else:
            burden_fig = px.histogram(
                df,
                x="mutation_burden",
                nbins=20,
                title="Overall mutation burden",
            )
            burden_fig.update_layout(
                xaxis_title="Number of mutated genes",
                yaxis_title="Number of patients",
                template="plotly_white",
                height=320,
                margin=dict(l=40, r=10, t=60, b=60),
            )

    # Survival by mutation
    mut_surv = make_km_by_mutation(df, col)

    return (
        summary_html,
        mut_pie,
        pam_fig,
        er_fig,
        treat_fig,
        burden_fig,
        mut_surv,
    )


# -------------------------------------------------
# CO-OCCURRENCE / CO-EXPRESSION TAB (OPTIMIZED)
# -------------------------------------------------


@app.callback(
    Output("co-heatmap", "figure"),
    Output("co-mut-network", "figure"),
    Output("co-mrna-heatmap", "figure"),
    Output("co-network", "figure"),
    Input("co-top-n", "value"),
    Input("co-mrna-topn", "value"),
)
def update_co_tab(top_n, mrna_topn):
    # default placeholders
    mut_heat = go.Figure().update_layout(
        title="No mutation columns",
        template="plotly_white",
        height=400,
    )
    mut_net = go.Figure().update_layout(
        title="Mutation network will appear once mutation genes are computed.",
        template="plotly_white",
        height=480,
    )
    mrna_heat = go.Figure().update_layout(
        title="No mRNA columns",
        template="plotly_white",
        height=400,
    )
    mrna_net = go.Figure().update_layout(
        title="mRNA co-expression network will appear once genes are computed.",
        template="plotly_white",
        height=480,
    )

    # ---------- MUTATION: co-mutation heatmap + network ----------
    if MUT_BIN is not None and MUT_COUNTS is not None and MUT_CORR_ALL is not None:
        counts = MUT_COUNTS
        mut_bin = MUT_BIN

        if top_n is None:
            top_n = 15
        top_n = int(max(5, min(top_n, min(30, len(counts)))))

        top_genes = counts.head(top_n).index.tolist()
        corr_mut = MUT_CORR_ALL.loc[top_genes, top_genes]

        # Heatmap
        mut_heat = px.imshow(
            corr_mut,
            x=[g.replace("_mut", "").upper() for g in corr_mut.columns],
            y=[g.replace("_mut", "").upper() for g in corr_mut.index],
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
            title=(
                f"Co-mutation correlation for top {top_n} mutated genes"
                "<br><sup>Cells show Pearson correlation of binary mutation presence (0/1).</sup>"
            ),
        )
        mut_heat.update_layout(
            template="plotly_white",
            height=480,
            margin=dict(l=80, r=40, t=80, b=80),
        )
        mut_heat.update_xaxes(tickangle=-45)

        # Network (positive + negative)
        genes_clean = [g.replace("_mut", "").upper() for g in top_genes]
        n = len(genes_clean)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        node_x = np.cos(angles)
        node_y = np.sin(angles)

        thr_mut = 0.1  # |corr| >= thr_mut -> edge
        pos_edge_x, pos_edge_y = [], []
        neg_edge_x, neg_edge_y = [], []

        for i in range(n):
            for j in range(i + 1, n):
                val = corr_mut.iloc[i, j]
                if val >= thr_mut:
                    pos_edge_x += [node_x[i], node_x[j], None]
                    pos_edge_y += [node_y[i], node_y[j], None]
                elif val <= -thr_mut:
                    neg_edge_x += [node_x[i], node_x[j], None]
                    neg_edge_y += [node_y[i], node_y[j], None]

        node_trace_mut = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=genes_clean,
            textposition="top center",
            marker=dict(
                size=16,
                color="cornflowerblue",
                line=dict(width=2, color="white"),
            ),
            hoverinfo="text",
        )

        edge_pos_trace = go.Scatter(
            x=pos_edge_x,
            y=pos_edge_y,
            mode="lines",
            line=dict(width=1, color="grey"),
            hoverinfo="none",
            showlegend=False,
        )
        edge_neg_trace = go.Scatter(
            x=neg_edge_x,
            y=neg_edge_y,
            mode="lines",
            line=dict(width=1, color="red", dash="dot"),
            hoverinfo="none",
            showlegend=False,
        )

        mut_net = go.Figure(
            data=[edge_pos_trace, edge_neg_trace, node_trace_mut]
        )
        mut_net.update_layout(
            title=(
                f"Co-mutation network for top {top_n} mutated genes"
                f"<br><sup>Nodes are genes; an edge connects two genes if |correlation| ≥ {thr_mut:.1f}."
                " Solid grey = positive, red dotted = negative.</sup>"
            ),
            template="plotly_white",
            height=480,
            margin=dict(l=20, r=20, t=80, b=20),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.05,
                xanchor="center",
                x=0.5,
            ),
        )

    # ---------- mRNA: co-expression heatmap + network ----------
    if MRNA_MATRIX is not None and MRNA_CORR_ALL is not None and MRNA_VAR is not None:
        X = MRNA_MATRIX

        if mrna_topn is None:
            mrna_topn = 20
        n_available = X.shape[1]
        mrna_topn = int(max(5, min(mrna_topn, min(50, n_available))))

        variances = MRNA_VAR
        top_genes_mrna = variances.head(mrna_topn).index.tolist()

        corr_mrna = MRNA_CORR_ALL.loc[top_genes_mrna, top_genes_mrna]

        # Heatmap
        mrna_heat = px.imshow(
            corr_mrna,
            x=top_genes_mrna,
            y=top_genes_mrna,
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
            title=(
                f"Co-expression of top {mrna_topn} most variable mRNA genes"
                "<br><sup>Cells show Pearson correlation of mRNA z-scores.</sup>"
            ),
        )
        mrna_heat.update_layout(
            template="plotly_white",
            height=480,
            margin=dict(l=80, r=40, t=80, b=80),
        )
        mrna_heat.update_xaxes(tickangle=-45)

        # Network (positive + negative)
        genes_clean_mrna = top_genes_mrna
        n_m = len(genes_clean_mrna)
        angles_m = np.linspace(0, 2 * np.pi, n_m, endpoint=False)
        node_x_m = np.cos(angles_m)
        node_y_m = np.sin(angles_m)

        thr = 0.3  # |corr| ≥ thr → edge
        pos_edge_x_m, pos_edge_y_m = [], []
        neg_edge_x_m, neg_edge_y_m = [], []

        for i in range(n_m):
            for j in range(i + 1, n_m):
                val = corr_mrna.iloc[i, j]
                if val >= thr:
                    pos_edge_x_m += [node_x_m[i], node_x_m[j], None]
                    pos_edge_y_m += [node_y_m[i], node_y_m[j], None]
                elif val <= -thr:
                    neg_edge_x_m += [node_x_m[i], node_x_m[j], None]
                    neg_edge_y_m += [node_y_m[i], node_y_m[j], None]

        node_deg = (
            (np.abs(corr_mrna.values) >= thr).sum(axis=0) - 1
        )  # exclude self

        node_trace_mrna = go.Scatter(
            x=node_x_m,
            y=node_y_m,
            mode="markers+text",
            text=genes_clean_mrna,
            textposition="top center",
            marker=dict(
                size=14,
                color=node_deg,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(
                    title=f"Node degree<br>(|corr| ≥ {thr:.1f})"
                ),
            ),
            customdata=node_deg,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Degree: %{customdata}<br>"
                "<extra></extra>"
            ),
            showlegend=False,
        )

        edge_pos_trace_m = go.Scatter(
            x=pos_edge_x_m,
            y=pos_edge_y_m,
            mode="lines",
            line=dict(width=1, color="grey"),
            hoverinfo="none",
            showlegend=False,
        )
        edge_neg_trace_m = go.Scatter(
            x=neg_edge_x_m,
            y=neg_edge_y_m,
            mode="lines",
            line=dict(width=1, color="red", dash="dot"),
            hoverinfo="none",
            showlegend=False,
        )

        mrna_net = go.Figure(
            data=[edge_pos_trace_m, edge_neg_trace_m, node_trace_mrna]
        )
        mrna_net.update_layout(
            title=(
                f"Co-expression network for top {mrna_topn} most variable mRNA genes"
                f"<br><sup>Nodes are gene expressions; an edge connects two genes if |correlation| ≥ {thr:.1f}."
                " Solid grey = positive, red dotted = negative.</sup>"
            ),
            template="plotly_white",
            height=480,
            margin=dict(l=20, r=20, t=80, b=20),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.05,
                xanchor="center",
                x=0.5,
            ),
        )

    return mut_heat, mut_net, mrna_heat, mrna_net


# -------------------------------------------------
# RUN
# -------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
