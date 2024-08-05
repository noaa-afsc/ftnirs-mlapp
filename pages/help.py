import dash
from dash import html


dash.register_page(__name__, path='/help')

layout = html.Div([
    html.H1('This is our Help page'),
    html.Div('This is our help page content.'),
])