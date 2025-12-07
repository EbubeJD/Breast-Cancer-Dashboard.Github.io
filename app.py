from dash import Dash

from layout import create_layout
from callbacks.overview import register_overview_callbacks
from callbacks.mrna import register_mrna_callbacks
from callbacks.mutation import register_mutation_callbacks
from callbacks.co import register_co_callbacks


app = Dash(__name__)
server = app.server
app.title = "METABRIC Breast Cancer Dashboard"

# Layout
app.layout = create_layout()

# Callbacks
register_overview_callbacks(app)
register_mrna_callbacks(app)
register_mutation_callbacks(app)
register_co_callbacks(app)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8050,
        debug=True,
    )
