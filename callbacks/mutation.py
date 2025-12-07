import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output

from data_config import df, mutation_cols, mrna_cols, SENTINEL_ZERO, make_km_for_subset


def register_mutation_callbacks(app):
    @app.callback(
        Output("mut-summary", "children"),
        Output("mut-pie", "figure"),
        Output("mut-by-pam50", "figure"),
        Output("mut-by-er", "figure"),
        Output("mut-treatment", "figure"),
        Output("mut-burden", "figure"),
        Output("mut-survival", "figure"),
        Output("mut-clinical-freq", "figure"),
        Output("mut-cofocus", "figure"),
        Output("mut-vs-mrna", "figure"),
        Input("mut-gene-dropdown", "value"),
        Input("mut-subtype-filter", "value"),
        Input("mut-er-filter", "value"),
        Input("mut-mrna-gene", "value"),
        Input("mut-cat-feature", "value"),
    )
    def update_mutation_tab(mut_col, subtype_filter, er_filter, mrna_gene, cat_feature):
        empty_fig = go.Figure().update_layout(title="No data", template="plotly_white", height=320)

        if not mut_col or mut_col not in df.columns:
            return [], empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig

        sub_df = df.copy()
        if subtype_filter and "pam50_+_claudin-low_subtype" in sub_df.columns:
            sub_df = sub_df[sub_df["pam50_+_claudin-low_subtype"].isin(subtype_filter)]
        if er_filter and "er_status" in sub_df.columns:
            sub_df = sub_df[sub_df["er_status"].isin(er_filter)]

        col_str = sub_df[mut_col].astype(str).str.strip()
        mutated_mask = ~col_str.isin(SENTINEL_ZERO)

        n_total = len(sub_df)
        n_mut = mutated_mask.sum()
        pct = (n_mut / n_total * 100).round(1) if n_total > 0 else 0

        from dash import html  # local import to avoid circular deps

        summary = [
            html.Div(
                style={"backgroundColor": "white", "padding": "12px 16px", "borderRadius": "8px", "boxShadow": "0 1px 2px rgba(0,0,0,0.08)"},
                children=[html.Div("Mutation", style={"fontSize": "12px", "color": "#666"}), html.Div(mut_col, style={"fontSize": "16px", "fontWeight": "bold"})],
            ),
            html.Div(
                style={"backgroundColor": "white", "padding": "12px 16px", "borderRadius": "8px", "boxShadow": "0 1px 2px rgba(0,0,0,0.08)"},
                children=[html.Div("Mutated patients (filtered)", style={"fontSize": "12px", "color": "#666"}), html.Div(f"{n_mut} / {n_total} ({pct}%)", style={"fontSize": "16px", "fontWeight": "bold"})],
            ),
        ]

        pie_fig = px.pie(
            names=["Mutated", "Not mutated"],
            values=[n_mut, n_total - n_mut],
            title=f"{mut_col} – prevalence in filtered cohort",
        )
        pie_fig.update_traces(textinfo="percent+label")
        pie_fig.update_layout(
            template="plotly_white",
            height=320,
            margin=dict(l=40, r=40, t=60, b=20),
        )

        # PAM50
        if "pam50_+_claudin-low_subtype" in sub_df.columns:
            tmp = sub_df[["pam50_+_claudin-low_subtype"]].copy()
            tmp["mutated"] = mutated_mask.astype(int)
            pam_group = tmp.groupby("pam50_+_claudin-low_subtype")["mutated"].agg(["sum", "count"]).reset_index()
            pam_group["percent_mutated"] = (pam_group["sum"] / pam_group["count"] * 100).round(1)
            pam_fig = px.bar(
                pam_group,
                x="pam50_+_claudin-low_subtype",
                y="percent_mutated",
                title=f"{mut_col} – % mutated by PAM50 (filtered)",
                text="percent_mutated",
            )
            pam_fig.update_traces(textposition="outside")
            max_y = None
            for _tr in pam_fig.data:
                y = getattr(_tr, "y", None)
                if y is None:
                    continue
                try:
                    this_max = max(y)
                except Exception:
                    continue
                max_y = this_max if max_y is None else max(max_y, this_max)
            if max_y is not None:
                pam_fig.update_yaxes(range=[0, max_y * 1.25])
            pam_fig.update_layout(
                xaxis_title="PAM50/Claudin-low subtype",
                yaxis_title="% of patients with mutation",
                template="plotly_white",
                height=320,
                margin=dict(l=60, r=20, t=60, b=80),
            )
        else:
            pam_fig = empty_fig.update_layout(title="No PAM50 column")

        # ER status
        if "er_status" in sub_df.columns:
            tmp2 = sub_df[["er_status"]].copy()
            tmp2["mutated"] = mutated_mask.astype(int)
            er_group = tmp2.groupby("er_status")["mutated"].agg(["sum", "count"]).reset_index()
            er_group["percent_mutated"] = (er_group["sum"] / er_group["count"] * 100).round(1)
            er_fig = px.bar(
                er_group,
                x="er_status",
                y="percent_mutated",
                title=f"{mut_col} – % mutated by ER status (filtered)",
                text="percent_mutated",
            )
            er_fig.update_traces(textposition="outside")
            max_y = None
            for _tr in er_fig.data:
                y = getattr(_tr, "y", None)
                if y is None:
                    continue
                try:
                    this_max = max(y)
                except Exception:
                    continue
                max_y = this_max if max_y is None else max(max_y, this_max)
            if max_y is not None:
                er_fig.update_yaxes(range=[0, max_y * 1.25])
            er_fig.update_layout(
                xaxis_title="ER status",
                yaxis_title="% of patients with mutation",
                template="plotly_white",
                height=320,
                margin=dict(l=60, r=20, t=60, b=60),
            )
        else:
            er_fig = empty_fig.update_layout(title="No ER column")

        # Treatment: percent of patients receiving each treatment (Yes only),
        # split by mutation status (Mutated vs Wild-type).
        treatment_cols = [c for c in ["chemotherapy", "hormone_therapy", "radio_therapy"] if c in sub_df.columns]

        if treatment_cols and n_total > 0:
            yes_keys = {"1", "yes", "y", "true"}
            no_keys = {"0", "no", "n", "false"}

            def map_treat(v: str) -> str:
                v = str(v).lower().strip()
                if v in yes_keys:
                    return "Yes"
                if v in no_keys or v == "" or v == "nan":
                    return "No"
                # anything else unknown -> treat as No
                return "No"

            long_rows = []
            mut_status_arr = np.where(mutated_mask, "Mutated", "Wild-type")

            for col in treatment_cols:
                t_series = sub_df[col].astype(str).fillna("").apply(map_treat)
                for ms, ts in zip(mut_status_arr, t_series):
                    long_rows.append({
                        "Mutation status": ms,
                        "Treatment status": ts,
                        "Treatment type": col.replace("_", " ").title(),
                    })

            long_df = pd.DataFrame(long_rows)

            # Keep only "Yes" and compute percentages within each mutation group
            yes_df = long_df[long_df["Treatment status"] == "Yes"].copy()

            # Counts of "Yes" per (Mutation status, Treatment type)
            grp_yes = (
                yes_df
                .groupby(["Mutation status", "Treatment type"])
                .size()
                .reset_index(name="CountYes")
            )

            # Denominator: total patients in each mutation group
            n_mut_group = mut_status_arr.tolist().count("Mutated")
            n_wt_group = mut_status_arr.tolist().count("Wild-type")

            def denom(row):
                return n_mut_group if row["Mutation status"] == "Mutated" else n_wt_group

            grp_yes["TotalGroup"] = grp_yes.apply(denom, axis=1)
            grp_yes["Percent"] = np.where(
                grp_yes["TotalGroup"] > 0,
                grp_yes["CountYes"] / grp_yes["TotalGroup"] * 100.0,
                0.0,
            )

            treat_fig = px.bar(
                grp_yes,
                x="Treatment type",
                y="Percent",
                color="Mutation status",
                barmode="group",
                title=(
                    f"Percent of patients receiving each treatment"
                ),
                text=grp_yes["Percent"].round(1).astype(str) + "%",
                category_orders={
                    "Mutation status": ["Wild-type", "Mutated"],
                },
            )
            treat_fig.update_traces(textposition="outside")
            max_y = None
            for _tr in treat_fig.data:
                y = getattr(_tr, "y", None)
                if y is None:
                    continue
                try:
                    this_max = max(y)
                except Exception:
                    continue
                max_y = this_max if max_y is None else max(max_y, this_max)
            if max_y is not None:
                treat_fig.update_yaxes(range=[0, max_y * 1.25])
            treat_fig.update_layout(
                xaxis_title="Treatment type",
                yaxis_title="% of patients in group",
                template="plotly_white",
                height=320,
                margin=dict(l=60, r=20, t=60, b=80),
                legend_title_text="Mutation status",
            )
        else:
            treat_fig = empty_fig.update_layout(title="No treatment data for filtered cohort")

        # Mutation burden
        if "mutation_count" in sub_df.columns:
            burden_df = sub_df.copy()
            burden_df["Mutation status"] = np.where(mutated_mask, "Mutated", "Wild-type")
            burden_fig = px.histogram(
                burden_df,
                x="mutation_count",
                color="Mutation status",
                barmode="overlay",
                nbins=30,
                title="Mutation count distribution (filtered)",
            )
            burden_fig.update_layout(
                xaxis_title="Mutation count",
                yaxis_title="Number of patients",
                template="plotly_white",
                height=320,
                margin=dict(l=60, r=20, t=60, b=60),
            )
        else:
            burden_fig = empty_fig.update_layout(title="mutation_count not found")

        # KM survival – filtered cohort
        if len(sub_df) > 0:
            surv_fig = make_km_for_subset(sub_df, mutated_mask, title_suffix=f"for {mut_col}")
        else:
            surv_fig = empty_fig.update_layout(title="No patients available for KM after filters")

        # Clinical feature % mutated
        if cat_feature and cat_feature in sub_df.columns and n_total > 0:
            cat_df = sub_df[[cat_feature]].copy()
            cat_df["mutated"] = mutated_mask.astype(int)
            grp = cat_df.groupby(cat_feature)["mutated"].agg(["sum", "count"]).reset_index()
            grp["pct"] = (grp["sum"] / grp["count"] * 100).round(1)
            clin_fig = px.bar(
                grp,
                x=cat_feature,
                y="pct",
                title=f"% {mut_col} mutated by {cat_feature.replace('_',' ').title()} (filtered)",
                text="pct"
            )
            clin_fig.update_traces(textposition="outside")
            max_y = None
            for _tr in clin_fig.data:
                y = getattr(_tr, "y", None)
                if y is None:
                    continue
                try:
                    this_max = max(y)
                except Exception:
                    continue
                max_y = this_max if max_y is None else max(max_y, this_max)
            if max_y is not None:
                clin_fig.update_yaxes(range=[0, max_y * 1.25])
            clin_fig.update_layout(
                xaxis_title=cat_feature.replace("_", " ").title(),
                yaxis_title="% of patients with mutation",
                template="plotly_white",
                height=280,
                margin=dict(l=60, r=20, t=60, b=80),
            )
        else:
            clin_fig = empty_fig.update_layout(title="Select a clinical feature")

        # Co-mutation focus: which other genes co-occur with this mutation?
        if n_mut > 0 and len(mutation_cols) > 1:
            base_subset = sub_df[mutated_mask]
            co_rows = []
            for g in mutation_cols:
                if g == mut_col or g not in base_subset.columns:
                    continue
                colg = base_subset[g].astype(str).str.strip()
                g_mut = ~colg.isin(SENTINEL_ZERO)
                total_g_mut = g_mut.sum()
                if total_g_mut == 0:
                    continue
                pct_co = (total_g_mut / n_mut) * 100.0
                co_rows.append({
                    "Gene": g.replace("_mut", "").upper(),
                    "Percent": round(pct_co, 1),
                })
            if co_rows:
                co_df = pd.DataFrame(co_rows).sort_values("Percent", ascending=False).head(10)
                cofocus_fig = px.bar(
                    co_df,
                    x="Gene",
                    y="Percent",
                    title=f"Top co-mutated genes with {mut_col}",
                    text="Percent",
                )
                cofocus_fig.update_traces(textposition="outside")
                max_y = None
                for _tr in cofocus_fig.data:
                    y = getattr(_tr, "y", None)
                    if y is None:
                        continue
                    try:
                        this_max = max(y)
                    except Exception:
                        continue
                    max_y = this_max if max_y is None else max(max_y, this_max)
                if max_y is not None:
                    cofocus_fig.update_yaxes(range=[0, max_y * 1.25])
                cofocus_fig.update_layout(
                    xaxis_title="Gene",
                    yaxis_title=f"% of patients also mutated",
                    template="plotly_white",
                    height=320,
                    margin=dict(l=60, r=20, t=60, b=60),
                )
            else:
                cofocus_fig = empty_fig.update_layout(title=f"No co-mutations found for {mut_col}")
        else:
            cofocus_fig = empty_fig.update_layout(title="No mutated patients for co-mutation analysis")

        # mRNA vs mutation
        if mrna_gene and mrna_gene in sub_df.columns:
            expr_df = pd.DataFrame({
                "Mutation status": np.where(mutated_mask, "Mutated", "Wild-type"),
                "Expression": sub_df[mrna_gene]
            }).dropna()
            if not expr_df.empty:
                expr_fig = px.box(
                    expr_df,
                    x="Mutation status",
                    y="Expression",
                    color="Mutation status",
                    title=f"{mrna_gene} expression vs {mut_col} status (filtered)",
                )
                expr_fig.update_layout(
                    xaxis_title="Mutation status",
                    yaxis_title=f"{mrna_gene} mRNA z-score",
                    template="plotly_white",
                    height=340,
                    margin=dict(l=60, r=20, t=60, b=60),
                )
            else:
                expr_fig = empty_fig.update_layout(title="No expression data after filtering")
        else:
            expr_fig = empty_fig.update_layout(title="Select an mRNA gene")

        return summary, pie_fig, pam_fig, er_fig, treat_fig, burden_fig, surv_fig, clin_fig, cofocus_fig, expr_fig
