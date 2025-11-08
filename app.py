from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
from thematic_dash import get_thematic_layout, register_thematic_callbacks
from security_dashboard import get_security_layout, register_security_callbacks

app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], suppress_callback_exceptions=True)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dbc.Container([
        dcc.Tabs(id="tabs", value='tab-thematic', children=[
            dcc.Tab(label='Thematic Dashboard', value='tab-thematic'),
            dcc.Tab(label='Security Dashboard', value='tab-security')
        ]),
        html.Div(id='tabs-content', style={'marginTop': '1em'})
    ], fluid=True),
])

@app.callback(Output('tabs-content', 'children'), Input('tabs', 'value'))
def render_tab_content(tab):
    if tab == 'tab-thematic':
        return get_thematic_layout()
    elif tab == 'tab-security':
        return get_security_layout()

# Register callbacks from both dashboards
register_thematic_callbacks(app)
register_security_callbacks(app)

if __name__ == "__main__":
    app.run(debug=True, port=8052)
