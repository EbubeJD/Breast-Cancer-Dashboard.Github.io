import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from data_config import df, mrna_cols


def register_mrna_callbacks(app):
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
                xaxis_title="PAM50/Claudin-low subtype",
                yaxis_title=f"{gene} mRNA z-score",
                template="plotly_white",
                height=340,
                margin=dict(l=60, r=20, t=60, b=80),
            )
        else:
            expr_fig = px.histogram(
                df,
                x=gene,
                nbins=40,
                title=f"Distribution of {gene}",
            )
            expr_fig.update_layout(
                xaxis_title=f"{gene} mRNA z-score",
                yaxis_title="Number of patients",
                template="plotly_white",
                height=340,
                margin=dict(l=60, r=20, t=60, b=60),
            )

        # Numeric mRNA matrix
        X = df[mrna_cols].apply(pd.to_numeric, errors="coerce")
        X = X.dropna(axis=1, how="all")
        X = X.fillna(X.median())

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

        # PCA + K-means
        n_features = X.shape[1]
        if n_components is None:
            n_components = 5
        n_components = int(max(2, min(n_components, min(10, n_features))))

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=n_components, random_state=0)
        X_pca = pca.fit_transform(X_scaled)

        pca_df = pd.DataFrame({
            "PC1": X_pca[:, 0],
            "PC2": X_pca[:, 1],
        })
        if "pam50_+_claudin-low_subtype" in df.columns:
            pca_df["Subtype"] = df["pam50_+_claudin-low_subtype"].fillna("Unknown")
        else:
            pca_df["Subtype"] = "All"

        pc1_var = pca.explained_variance_ratio_[0] * 100
        pc2_var = pca.explained_variance_ratio_[1] * 100

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
            y=pca.explained_variance_ratio_,
            labels={"x": "Principal component", "y": "Explained variance ratio"},
            title="Explained variance by principal component",
        )
        var_fig.update_layout(
            template="plotly_white",
            height=340,
            margin=dict(l=60, r=20, t=60, b=60),
        )

        n_samples = X_pca.shape[0]
        if k is None:
            k = 3
        k = int(max(2, min(int(k), 8)))

        if n_samples <= k:
            kmeans_fig = empty_fig("Not enough samples for chosen k.")
            summary_text = (
                f"Cannot run K-means with k={k} when number of samples={n_samples}."
            )
            return expr_fig, corr_fig, pca_scatter, var_fig, kmeans_fig, summary_text

        k_dim = min(5, n_components)
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
        cluster_labels = kmeans.fit_predict(X_pca[:, :k_dim])

        try:
            sil = silhouette_score(X_pca[:, :k_dim], cluster_labels)
            sil_text = f"Silhouette score: {sil:.3f}"
        except Exception:
            sil_text = "Silhouette score could not be computed."

        pca_df["Cluster"] = cluster_labels.astype(str)

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
            comp = (
                pca_df.groupby(["Cluster", "Subtype"])
                .size()
                .reset_index(name="count")
            )
            for cl in sorted(pca_df["Cluster"].unique(), key=lambda x: int(x)):
                sub_df2 = comp[comp["Cluster"] == cl]
                pieces = [
                    f"{row['Subtype']}: {row['count']}"
                    for _, row in sub_df2.iterrows()
                ]
                lines.append(f"Cluster {cl}: " + ", ".join(pieces))

        from dash import html  # imported here to avoid circular issues

        summary_text = html.Div([
            html.Div(sil_text, style={"fontWeight": "bold", "marginBottom": "8px"}),
            html.Div("K-means clustering is performed on the PCA-transformed mRNA matrix."),
            html.Ul([
                html.Li(f"Number of PCA components used: {n_components}"),
                html.Li(f"Number of clusters (k): {k}"),
            ]),
            html.Div("Cluster composition by subtype:", style={"marginTop": "8px", "fontWeight": "bold"}),
            html.Pre("\n".join(lines) if lines else "Subtype labels not available."),
        ])

        return expr_fig, corr_fig, pca_scatter, var_fig, kmeans_fig, summary_text
