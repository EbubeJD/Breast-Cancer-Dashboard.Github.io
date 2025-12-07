import numpy as np
import pandas as pd
import plotly.express as px
from dash import html, dcc
from dash.dependencies import Input, Output

from data_config import (
    df,
    clinical_cols,
    mutation_cols,
    mrna_cols,
    SENTINEL_ZERO,
    make_km_figure,
)
from layout import stat_card


def register_overview_callbacks(app):
    @app.callback(
        Output("tab-content", "children"),
        Input("tabs", "value")
    )
    def render_content(tab):
        # ------------------ OVERVIEW TAB ------------------
        if tab == "tab-overview":
            n_patients = df.shape[0]
            median_age = df["age_at_diagnosis"].median() if "age_at_diagnosis" in df.columns else None

            er_pos = None
            if "er_status" in df.columns:
                er_pos = (df["er_status"].astype(str).str.upper() == "POSITIVE").mean() * 100

            alive_pct = None
            if "overall_survival" in df.columns:
                vals = df["overall_survival"].astype(str).str.lower().str.strip()
                alive_keywords = ["alive", "living", "yes", "1", "true"]
                dead_keywords = ["dead", "died", "no", "0", "false"]
                alive_mask = vals.apply(lambda v: any(k in v for k in alive_keywords))
                dead_mask = vals.apply(lambda v: any(k in v for k in dead_keywords))
                if alive_mask.any() or dead_mask.any():
                    alive_pct = 100 * alive_mask.sum() / (alive_mask.sum() + dead_mask.sum())

            # Age histogram
            if "age_at_diagnosis" in df.columns:
                age_fig = px.histogram(
                    df,
                    x="age_at_diagnosis",
                    nbins=25,
                    title="Age at Diagnosis",
                )
                age_fig.update_layout(
                    xaxis_title="Age at diagnosis (years)",
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
                        return np.nan
                    try:
                        x = float(v)
                    except Exception:
                        return np.nan

                    # Bin into 0–4
                    if x < 0.5:
                        s = 0
                    elif x < 1.5:
                        s = 1
                    elif x < 2.5:
                        s = 2
                    elif x < 3.5:
                        s = 3
                    else:
                        s = 4
                    return f"Stage {s}"

                stage_df["tumor_stage_label"] = stage_df["tumor_stage"].apply(_stage_label)

                stage_order = ["Stage 0", "Stage 1", "Stage 2", "Stage 3", "Stage 4"]

                stage_fig = px.histogram(
                    stage_df,
                    x="tumor_stage_label",
                    title="Tumor Stage",
                    category_orders={"tumor_stage_label": stage_order},
                )
                stage_fig.update_layout(
                    xaxis_title="Tumor stage",
                    yaxis_title="Number of patients",
                    template="plotly_white",
                    height=340,
                    margin=dict(l=40, r=10, t=60, b=60),
                )
            else:
                stage_fig = None

            # Surgery pie
            if "type_of_breast_surgery" in df.columns:
                df["type_of_breast_surgery"] = (
                    df["type_of_breast_surgery"]
                    .fillna("None")
                    .replace("", "None")
                    .replace(" ", "None")
                )

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

            # KM survival figure
            km_fig = make_km_figure(df)

            # Nottingham
            if "nottingham_prognostic_index" in df.columns:
                npi_fig = px.histogram(
                    df,
                    x="nottingham_prognostic_index",
                    nbins=30,
                    title=(
                        "Nottingham Prognostic Index<br>"
                        "<sup>Combines tumor size, lymph-node involvement, and grade; "
                        "higher values generally indicate worse prognosis.</sup>"
                    ),
                )
                npi_fig.update_layout(
                    xaxis_title="Nottingham Prognostic Index",
                    yaxis_title="Number of patients",
                    template="plotly_white",
                    height=360,
                    margin=dict(l=40, r=10, t=70, b=60),
                )
            else:
                npi_fig = None

            # Top mutations + heatmap
            mut_fig = None
            heat_fig = None
            if mutation_cols:
                mut_str = df[mutation_cols].astype(str).apply(lambda s: s.str.strip())
                mut_bin = (~mut_str.isin(SENTINEL_ZERO)).astype(int)
                mutation_counts_clean = mut_bin.sum().sort_values(ascending=False)
                top10_cols = mutation_counts_clean.head(10).index.tolist()
                top10 = mutation_counts_clean.head(10).reset_index()
                top10.columns = ["Gene", "Patients_Mutated"]
                top10["Gene"] = top10["Gene"].str.replace("_mut", "", regex=False).str.upper()

                mut_fig = px.bar(
                    top10,
                    x="Gene",
                    y="Patients_Mutated",
                    title="Top 10 Most Frequently Mutated Genes in METABRIC (binary presence)",
                    text="Patients_Mutated",
                    color="Patients_Mutated",
                    color_continuous_scale="Blues",
                )
                mut_fig.update_traces(textposition="outside")
                max_y = max(top10["Patients_Mutated"]) if not top10.empty else 0
                if max_y > 0:
                    mut_fig.update_yaxes(range=[0, max_y * 1.25])
                mut_fig.update_layout(
                    yaxis_title="Number of patients with mutation",
                    xaxis_title="Gene",
                    xaxis_tickangle=-35,
                    template="plotly_white",
                    height=380,
                    margin=dict(l=40, r=20, t=70, b=80),
                    coloraxis_colorbar_title="Count",
                )

                if "pam50_+_claudin-low_subtype" in df.columns:
                    sub_col = "pam50_+_claudin-low_subtype"
                    sub_df = df[[sub_col] + top10_cols].copy()
                    sub_df[sub_col] = sub_df[sub_col].fillna("Unknown")
                    subtype_order = sorted(sub_df[sub_col].unique())
                    heat_pct = pd.DataFrame(index=subtype_order)
                    for orig_col in top10_cols:
                        col_clean = orig_col.replace("_mut", "").upper()
                        col_str = sub_df[orig_col].astype(str).str.strip()
                        mutated = ~col_str.isin(SENTINEL_ZERO)
                        grp = (
                            pd.DataFrame({"Subtype": sub_df[sub_col], "Mutated": mutated.astype(int)})
                            .groupby("Subtype")["Mutated"]
                            .agg(["sum", "count"])
                        )
                        grp["pct"] = (grp["sum"] / grp["count"] * 100).round(1)
                        heat_pct[col_clean] = grp["pct"].reindex(subtype_order).fillna(0)

                    heat_fig = px.imshow(
                        heat_pct.values,
                        labels=dict(
                            x="Gene",
                            y="PAM50/Claudin-low subtype",
                            color="% mutated",
                        ),
                        x=heat_pct.columns.tolist(),
                        y=heat_pct.index.tolist(),
                        text_auto=True,
                        aspect="auto",
                        title="Mutation rate (%) by subtype for top 10 mutated genes",
                        color_continuous_scale="Plasma",
                    )
                    heat_fig.update_layout(
                        xaxis_title="Gene",
                        yaxis_title="Subtype",
                        template="plotly_white",
                        height=550,
                        margin=dict(l=80, r=20, t=80, b=60),
                    )

            return html.Div(
                children=[
                    # summary cards
                    html.Div(
                        style={"display": "flex", "gap": "15px", "flexWrap": "wrap", "marginBottom": "25px"},
                        children=[
                            stat_card("Patients", f"{n_patients}"),
                            stat_card("Clinical attributes", f"{len(clinical_cols)}"),
                            stat_card("Mutation genes", f"{len(mutation_cols)}"),
                            stat_card("mRNA genes", f"{len(mrna_cols)}"),
                            stat_card("Median age", f"{median_age:.1f}" if median_age else "—"),
                            stat_card("ER+ (%)", f"{er_pos:.1f}%" if er_pos is not None else "—"),
                            stat_card("Alive (%)", f"{alive_pct:.1f}%" if alive_pct is not None else "—"),
                        ],
                    ),

                    # clinical row
                    html.Div(
                        style={"display": "flex", "gap": "25px", "flexWrap": "wrap", "marginBottom": "25px"},
                        children=[
                            html.Div(
                                style={"flex": "1 1 350px", "backgroundColor": "white", "padding": "10px", "borderRadius": "8px"},
                                children=[dcc.Graph(figure=age_fig)] if age_fig else ["age_at_diagnosis not found"],
                            ),
                            html.Div(
                                style={"flex": "1 1 350px", "backgroundColor": "white", "padding": "10px", "borderRadius": "8px"},
                                children=[dcc.Graph(figure=stage_fig)] if stage_fig else ["tumor_stage not found"],
                            ),
                            html.Div(
                                style={"flex": "1 1 350px", "backgroundColor": "white", "padding": "10px", "borderRadius": "8px"},
                                children=[dcc.Graph(figure=surgery_fig)] if surgery_fig else ["type_of_breast_surgery not found"],
                            ),
                        ],
                    ),

                    # survival row
                    html.Div(
                        style={"display": "flex", "gap": "25px", "flexWrap": "wrap", "marginBottom": "25px"},
                        children=[
                            html.Div(
                                style={"flex": "1 1 400px", "backgroundColor": "white", "padding": "10px", "borderRadius": "8px"},
                                children=[dcc.Graph(figure=km_fig)],
                            ),
                            html.Div(
                                style={"flex": "1 1 400px", "backgroundColor": "white", "padding": "10px", "borderRadius": "8px"},
                                children=[dcc.Graph(figure=npi_fig)] if npi_fig else ["nottingham_prognostic_index not found"],
                            ),
                        ],
                    ),

                    # mutation row
                    html.Div(
                        style={"display": "flex", "gap": "25px", "flexWrap": "wrap"},
                        children=[
                            html.Div(
                                style={"flex": "1 1 450px", "backgroundColor": "white", "padding": "10px", "borderRadius": "8px"},
                                children=[dcc.Graph(figure=mut_fig)] if mut_fig else ["No mutation columns detected."],
                            ),
                            html.Div(
                                style={"flex": "1 1 650px", "backgroundColor": "white", "padding": "10px", "borderRadius": "8px"},
                                children=[dcc.Graph(figure=heat_fig)] if heat_fig else [],
                            ),
                        ],
                    ),
                ]
            )

        # ------------------ MUTATION TAB ------------------
        elif tab == "tab-mutations":
            mut_options = [{"label": c, "value": c} for c in mutation_cols]
            mrna_options = [{"label": c, "value": c} for c in mrna_cols[:200]]
            cat_features = [
                c for c in ["tumor_stage", "type_of_breast_surgery", "inferred_menopausal_state"]
                if c in df.columns
            ]

            return html.Div([
                # Sticky global filters + summary
                html.Div(
                    style={
                        "position": "sticky",
                        "top": 0,
                        "zIndex": 5,
                        "backgroundColor": "#f5f5f5",
                        "paddingBottom": "10px",
                        "marginBottom": "15px",
                        "borderBottom": "1px solid #ddd",
                    },
                    children=[
                        html.H3("Mutation Analysis Workspace"),
                        html.Div(
                            style={
                                "display": "flex",
                                "gap": "15px",
                                "flexWrap": "wrap",
                                "marginBottom": "10px",
                            },
                            children=[
                                dcc.Dropdown(
                                    id="mut-gene-dropdown",
                                    options=mut_options,
                                    value=mutation_cols[0] if mutation_cols else None,
                                    style={"width": "260px"},
                                    placeholder="Select mutation...",
                                    clearable=False,
                                ),
                                dcc.Dropdown(
                                    id="mut-subtype-filter",
                                    options=(
                                        [{"label": s, "value": s}
                                         for s in sorted(df["pam50_+_claudin-low_subtype"].dropna().unique())]
                                        if "pam50_+_claudin-low_subtype" in df.columns else []
                                    ),
                                    multi=True,
                                    placeholder="Filter by PAM50 subtype…",
                                    style={"width": "240px"},
                                ),
                                dcc.Dropdown(
                                    id="mut-er-filter",
                                    options=(
                                        [{"label": s, "value": s}
                                         for s in sorted(df["er_status"].dropna().unique())]
                                        if "er_status" in df.columns else []
                                    ),
                                    multi=True,
                                    placeholder="Filter by ER status…",
                                    style={"width": "200px"},
                                ),
                            ],
                        ),
                        html.Div(
                            id="mut-summary",
                            style={
                                "display": "flex",
                                "gap": "15px",
                                "flexWrap": "wrap",
                            },
                        ),
                    ],
                ),

                # Row 1: prevalence + subtype + ER
                html.Div(
                    style={"display": "flex", "gap": "25px", "flexWrap": "wrap",
                           "marginBottom": "25px"},
                    children=[
                        html.Div(
                            style={
                                "flex": "1 1 300px",
                                "backgroundColor": "white",
                                "padding": "10px",
                                "borderRadius": "8px",
                            },
                            children=[dcc.Graph(id="mut-pie", style={"height": "320px"})],
                        ),
                        html.Div(
                            style={
                                "flex": "1 1 360px",
                                "backgroundColor": "white",
                                "padding": "10px",
                                "borderRadius": "8px",
                            },
                            children=[dcc.Graph(id="mut-by-pam50", style={"height": "320px"})],
                        ),
                        html.Div(
                            style={
                                "flex": "1 1 360px",
                                "backgroundColor": "white",
                                "padding": "10px",
                                "borderRadius": "8px",
                            },
                            children=[dcc.Graph(id="mut-by-er", style={"height": "320px"})],
                        ),
                    ],
                ),

                # Row 2: treatment + mutation burden + co-mutation focus
                html.Div(
                    style={"display": "flex", "gap": "25px", "flexWrap": "wrap",
                           "marginBottom": "25px"},
                    children=[
                        html.Div(
                            style={
                                "flex": "1 1 320px",
                                "backgroundColor": "white",
                                "padding": "10px",
                                "borderRadius": "8px",
                            },
                            children=[dcc.Graph(id="mut-treatment", style={"height": "320px"})],
                        ),
                        html.Div(
                            style={
                                "flex": "1 1 320px",
                                "backgroundColor": "white",
                                "padding": "10px",
                                "borderRadius": "8px",
                            },
                            children=[dcc.Graph(id="mut-burden", style={"height": "320px"})],
                        ),
                        html.Div(
                            style={
                                "flex": "1 1 320px",
                                "backgroundColor": "white",
                                "padding": "10px",
                                "borderRadius": "8px",
                            },
                            children=[dcc.Graph(id="mut-cofocus", style={"height": "320px"})],
                        ),
                    ],
                ),

                # Row 3: survival + clinical feature
                html.Div(
                    style={"display": "flex", "gap": "25px", "flexWrap": "wrap",
                           "marginBottom": "25px"},
                    children=[
                        html.Div(
                            style={
                                "flex": "1 1 400px",
                                "backgroundColor": "white",
                                "padding": "10px",
                                "borderRadius": "8px",
                                "height": "360px",
                                "overflow": "hidden",
                                "display": "flex",
                                "flexDirection": "column",
                            },
                            children=[
                                dcc.Graph(
                                    id="mut-survival",
                                    style={"height": "320px", "overflow": "hidden"},
                                    config={"displayModeBar": False},
                                ),
                            ],
                        ),
                        html.Div(
                            style={
                                "flex": "1 1 360px",
                                "backgroundColor": "white",
                                "padding": "12px",
                                "borderRadius": "8px",
                                "height": "360px",
                                "overflow": "hidden",
                                "display": "flex",
                                "flexDirection": "column",
                            },
                            children=[
                                html.Div(
                                    style={"marginBottom": "8px"},
                                    children=[
                                        html.Label(
                                            "Clinical feature for % mutated",
                                            style={"fontSize": "12px", "fontWeight": "bold"},
                                        ),
                                        dcc.Dropdown(
                                            id="mut-cat-feature",
                                            options=[
                                                {
                                                    "label": c.replace("_", " ").title(),
                                                    "value": c,
                                                }
                                                for c in cat_features
                                            ],
                                            value=cat_features[0] if cat_features else None,
                                            placeholder="Choose feature…",
                                            clearable=False,
                                            style={"marginTop": "4px"},
                                        ),
                                    ],
                                ),
                                dcc.Graph(
                                    id="mut-clinical-freq",
                                    style={
                                        "flex": "1 1 auto",
                                        "height": "280px",
                                        "overflow": "hidden",
                                    },
                                    config={"displayModeBar": False},
                                ),
                            ],
                        ),
                    ],
                ),

                # Row 4: mRNA vs mutation
                html.Div(
                    style={
                        "backgroundColor": "white",
                        "padding": "12px",
                        "borderRadius": "8px",
                    },
                    children=[
                        html.Div(
                            style={
                                "display": "flex",
                                "alignItems": "center",
                                "gap": "10px",
                                "marginBottom": "6px",
                            },
                            children=[
                                html.Label(
                                    "mRNA gene for expression vs mutation",
                                    style={"fontSize": "12px", "fontWeight": "bold"},
                                ),
                                dcc.Dropdown(
                                    id="mut-mrna-gene",
                                    options=mrna_options,
                                    value=mrna_cols[0] if mrna_cols else None,
                                    style={"width": "260px"},
                                    placeholder="Select mRNA gene…",
                                ),
                            ],
                        ),
                        dcc.Graph(id="mut-vs-mrna", style={"height": "340px"}),
                    ],
                ),
            ])

        # ------------------ mRNA TAB ------------------
        elif tab == "tab-mrna":
            options = [{"label": c, "value": c} for c in mrna_cols[:200]]

            return html.Div([

                # Sticky controls
                html.Div(
                    style={
                        "position": "sticky",
                        "top": 0,
                        "zIndex": 5,
                        "backgroundColor": "#f5f5f5",
                        "paddingBottom": "10px",
                        "marginBottom": "15px",
                        "borderBottom": "1px solid #ddd",
                    },
                    children=[
                        html.H3("mRNA Expression Explorer"),
                        html.Div(
                            style={"display": "flex", "gap": "15px", "flexWrap": "wrap"},
                            children=[
                                dcc.Dropdown(
                                    id="mrna-gene-dropdown",
                                    options=options,
                                    value=mrna_cols[0] if mrna_cols else None,
                                    style={"width": "320px"},
                                    placeholder="Select mRNA gene...",
                                ),
                                html.Div(style={"display": "flex", "flexDirection": "column"},
                                         children=[
                                             html.Label("Number of PCA components", style={"fontSize": "12px"}),
                                             dcc.Slider(
                                                 id="mrna-pca-components",
                                                 min=2,
                                                 max=10,
                                                 step=1,
                                                 value=5,
                                                 marks={2: "2", 5: "5", 10: "10"},
                                                 tooltip={"placement": "bottom"}
                                             ),
                                         ]),
                                html.Div(style={"display": "flex", "flexDirection": "column"},
                                         children=[
                                             html.Label("K-means clusters (k)", style={"fontSize": "12px"}),
                                             dcc.Slider(
                                                 id="mrna-k",
                                                 min=2,
                                                 max=8,
                                                 step=1,
                                                 value=3,
                                                 marks={2: "2", 3: "3", 5: "5", 8: "8"},
                                                 tooltip={"placement": "bottom"}
                                             ),
                                         ]),
                            ],
                        ),
                    ],
                ),

                # row 1: expression + correlation
                html.Div(
                    style={"display": "flex", "gap": "25px", "flexWrap": "wrap", "marginBottom": "25px"},
                    children=[
                        html.Div(
                            style={"flex": "1 1 380px", "backgroundColor": "white",
                                   "padding": "10px", "borderRadius": "8px"},
                            children=[dcc.Graph(id="mrna-gene-box", style={"height": "340px"})],
                        ),
                        html.Div(
                            style={"flex": "1 1 380px", "backgroundColor": "white",
                                   "padding": "10px", "borderRadius": "8px"},
                            children=[dcc.Graph(id="mrna-corr-bar", style={"height": "340px"})],
                        ),
                    ],
                ),

                # row 2: PCA scatter + explained variance
                html.Div(
                    style={"display": "flex", "gap": "25px", "flexWrap": "wrap", "marginBottom": "25px"},
                    children=[
                        html.Div(
                            style={"flex": "1 1 420px", "backgroundColor": "white",
                                   "padding": "10px", "borderRadius": "8px"},
                            children=[dcc.Graph(id="mrna-pca-scatter", style={"height": "340px"})],
                        ),
                        html.Div(
                            style={"flex": "1 1 380px", "backgroundColor": "white",
                                   "padding": "10px", "borderRadius": "8px"},
                            children=[dcc.Graph(id="mrna-pca-var", style={"height": "340px"})],
                        ),
                    ],
                ),

                # row 3: K-means clustering + summary
                html.Div(
                    style={"display": "flex", "gap": "25px", "flexWrap": "wrap"},
                    children=[
                        html.Div(
                            style={"flex": "1 1 420px", "backgroundColor": "white",
                                   "padding": "10px", "borderRadius": "8px"},
                            children=[dcc.Graph(id="mrna-kmeans-scatter", style={"height": "340px"})],
                        ),
                        html.Div(
                            id="mrna-kmeans-summary",
                            style={
                                "flex": "1 1 300px",
                                "backgroundColor": "white",
                                "padding": "15px",
                                "borderRadius": "8px",
                                "fontSize": "14px",
                                "lineHeight": "1.5",
                            },
                        ),
                    ],
                ),
            ])

        # ------------------ CO-MUTATION / CO-EXPRESSION TAB ------------------
        else:  # tab == "tab-comutation"
            return html.Div([
                # Sticky controls
                html.Div(
                    style={
                        "position": "sticky",
                        "top": 0,
                        "zIndex": 5,
                        "backgroundColor": "#f5f5f5",
                        "paddingBottom": "10px",
                        "marginBottom": "15px",
                        "borderBottom": "1px solid #ddd",
                    },
                    children=[
                        html.H3("Global Co-Mutation and mRNA Co-Expression"),
                        html.P(
                            "Explore how genes co-mutate at the DNA level and co-express at the mRNA level.",
                            style={"marginBottom": "6px"},
                        ),
                        html.Div(
                            style={"display": "flex", "gap": "20px", "flexWrap": "wrap"},
                            children=[
                                html.Div(
                                    style={"display": "flex", "flexDirection": "column"},
                                    children=[
                                        html.Label(
                                            "Top mutated genes (for co-mutation heatmap & network)",
                                            style={"fontSize": "12px"}
                                        ),
                                        dcc.Slider(
                                            id="co-top-n",
                                            min=5,
                                            max=30,
                                            step=1,
                                            value=15,
                                            marks={5: "5", 10: "10", 15: "15", 20: "20", 25: "25", 30: "30"},
                                            tooltip={"placement": "bottom"},
                                        ),
                                    ],
                                ),
                                html.Div(
                                    style={"display": "flex", "flexDirection": "column"},
                                    children=[
                                        html.Label(
                                            "Top-variable mRNA genes (for co-expression heatmap & network)",
                                            style={"fontSize": "12px"}
                                        ),
                                        dcc.Slider(
                                            id="co-mrna-topn",
                                            min=5,
                                            max=50,
                                            step=1,
                                            value=20,
                                            marks={5: "5", 20: "20", 50: "50"},
                                            tooltip={"placement": "bottom"},
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),

                # Row 1: mutation section
                html.Div(
                    style={"display": "flex", "gap": "20px", "flexWrap": "wrap", "marginBottom": "20px"},
                    children=[
                        html.Div(
                            style={"flex": "1 1 450px", "backgroundColor": "white", "padding": "10px", "borderRadius": "8px"},
                            children=[dcc.Graph(id="co-heatmap", style={"height": "480px"})],
                        ),
                        html.Div(
                            style={"flex": "1 1 450px", "backgroundColor": "white", "padding": "10px", "borderRadius": "8px"},
                            children=[dcc.Graph(id="co-mut-network", style={"height": "480px"})],
                        ),
                    ],
                ),

                # Row 2: mRNA section
                html.Div(
                    style={"display": "flex", "gap": "20px", "flexWrap": "wrap"},
                    children=[
                        html.Div(
                            style={"flex": "1 1 450px", "backgroundColor": "white", "padding": "10px", "borderRadius": "8px"},
                            children=[dcc.Graph(id="co-mrna-heatmap", style={"height": "480px"})],
                        ),
                        html.Div(
                            style={"flex": "1 1 450px", "backgroundColor": "white", "padding": "10px", "borderRadius": "8px"},
                            children=[dcc.Graph(id="co-network", style={"height": "480px"})],
                        ),
                    ],
                ),
            ])
