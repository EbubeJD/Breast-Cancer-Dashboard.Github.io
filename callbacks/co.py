import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output

from data_config import df, mutation_cols, mrna_cols, SENTINEL_ZERO


def register_co_callbacks(app):
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
        if mutation_cols:
            mut_str = df[mutation_cols].astype(str).apply(lambda s: s.str.strip())
            mut_bin = (~mut_str.isin(SENTINEL_ZERO)).astype(int)

            counts = mut_bin.sum().sort_values(ascending=False)
            if top_n is None:
                top_n = 15
            top_n = int(max(5, min(top_n, min(30, len(counts)))))

            top_genes = counts.head(top_n).index.tolist()
            corr_mut = mut_bin[top_genes].corr()

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

            edge_pos_trace = go.Scatter(
                x=pos_edge_x,
                y=pos_edge_y,
                mode="lines",
                line=dict(width=1.5, color="rgba(80, 80, 80, 0.7)"),
                hoverinfo="none",
                showlegend=True,
                name=f"Positive edge (≥ {thr_mut:.1f})",
            )
            edge_neg_trace = go.Scatter(
                x=neg_edge_x,
                y=neg_edge_y,
                mode="lines",
                line=dict(width=1.5, color="rgba(200, 60, 60, 0.85)", dash="dot"),
                hoverinfo="none",
                showlegend=True,
                name=f"Negative edge (≤ -{thr_mut:.1f})",
            )

            # degree by |corr|
            degrees = np.zeros(n, dtype=int)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    val = corr_mut.iloc[i, j]
                    if abs(val) >= thr_mut:
                        degrees[i] += 1

            node_trace_mut = go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                text=genes_clean,
                textposition="top center",
                customdata=degrees,
                marker=dict(
                    size=10 + 2 * degrees,
                    color=degrees,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(
                        title=f"Node degree<br>(|corr| ≥ {thr_mut:.1f})"
                    ),
                ),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Degree: %{customdata}<br>"
                    "<extra></extra>"
                ),
                showlegend=False,
            )

            mut_net = go.Figure(data=[edge_pos_trace, edge_neg_trace, node_trace_mut])
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
        if mrna_cols:
            import pandas as pd

            X = df[mrna_cols].apply(pd.to_numeric, errors="coerce")
            X = X.dropna(axis=1, how="all")
            X = X.fillna(X.median())

            if mrna_topn is None:
                mrna_topn = 20
            n_available = X.shape[1]
            mrna_topn = int(max(5, min(mrna_topn, min(50, n_available))))

            variances = X.var().sort_values(ascending=False)
            top_genes_mrna = variances.head(mrna_topn).index.tolist()

            corr_mrna = X[top_genes_mrna].corr()

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
                    "<br><sup>Cells show Pearson correlation between mRNA z-scores.</sup>"
                ),
                aspect="auto",
            )
            mrna_heat.update_layout(
                template="plotly_white",
                height=480,
                margin=dict(l=100, r=40, t=80, b=80),
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

            edge_pos_trace_m = go.Scatter(
                x=pos_edge_x_m,
                y=pos_edge_y_m,
                mode="lines",
                line=dict(width=1.5, color="rgba(80, 80, 80, 0.7)"),
                hoverinfo="none",
                showlegend=True,
                name=f"Positive edge (≥ {thr:.1f})",
            )
            edge_neg_trace_m = go.Scatter(
                x=neg_edge_x_m,
                y=neg_edge_y_m,
                mode="lines",
                line=dict(width=1.5, color="rgba(200, 60, 60, 0.85)", dash="dot"),
                hoverinfo="none",
                showlegend=True,
                name=f"Negative edge (≤ -{thr:.1f})",
            )

            degrees_m = np.zeros(n_m, dtype=int)
            for i in range(n_m):
                for j in range(n_m):
                    if i == j:
                        continue
                    val = corr_mrna.iloc[i, j]
                    if abs(val) >= thr:
                        degrees_m[i] += 1

            node_trace_mrna = go.Scatter(
                x=node_x_m,
                y=node_y_m,
                mode="markers+text",
                text=genes_clean_mrna,
                textposition="top center",
                customdata=degrees_m,
                marker=dict(
                    size=10 + 2 * degrees_m,
                    color=degrees_m,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(
                        title=f"Node degree<br>(|corr| ≥ {thr:.1f})"
                    ),
                ),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Degree: %{customdata}<br>"
                    "<extra></extra>"
                ),
                showlegend=False,
            )

            mrna_net = go.Figure(data=[edge_pos_trace_m, edge_neg_trace_m, node_trace_mrna])
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
