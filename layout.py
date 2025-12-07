from dash import html, dcc

def stat_card(title, value):
    return html.Div(
        style={
            "backgroundColor": "white",
            "padding": "15px 20px",
            "borderRadius": "8px",
            "boxShadow": "0 1px 2px rgba(0,0,0,0.08)",
            "minWidth": "180px",
        },
        children=[
            html.Div(title, style={"fontSize": "13px", "color": "#666"}),
            html.Div(value, style={"fontSize": "24px", "fontWeight": "bold"}),
        ],
    )


def create_layout():
    """Top-level layout with tabs and a container for tab content."""
    return html.Div(
        style={"fontFamily": "Arial", "backgroundColor": "#f5f5f5", "minHeight": "100vh"},
        children=[
            html.Div(
                style={"padding": "25px 25px 5px 25px"},
                children=[
                    html.H1("METABRIC Breast Cancer Dashboard", style={"marginBottom": "5px"}),
                    html.P(
                        "Explore clinical attributes, mutation profiles, and mRNA expression.",
                        style={"color": "#555"},
                    ),
                ],
            ),
            dcc.Tabs(
                id="tabs",
                value="tab-overview",
                children=[
                    dcc.Tab(label="Overview", value="tab-overview"),
                    dcc.Tab(label="Mutations", value="tab-mutations"),
                    dcc.Tab(label="mRNA Expression", value="tab-mrna"),
                    dcc.Tab(label="Co-Mutation / Co-Expression", value="tab-comutation"),
                ],
                style={"padding": "0 25px"},
            ),
            html.Div(id="tab-content", style={"padding": "25px"}),
        ],
    )
