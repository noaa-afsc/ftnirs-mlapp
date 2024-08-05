import dash
from dash import html
from app_name import app_name
from dotenv import load_dotenv
import os

load_dotenv('./tmp/.env')

dash.register_page(__name__, path="/help",name =f'{os.getenv("APPNAME")} Help')

layout = html.Div([
    html.H1('This is our Help page'),
    html.Div('This is our help page content.'),
])