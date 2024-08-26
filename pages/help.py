import dash
from click import style
from dash import html
from dotenv import load_dotenv
import os
from app_constant import app_header,header_height

load_dotenv('./tmp/.env')

dash.register_page(__name__, path="/help",name =f'{os.getenv("APPNAME")} Help')

layout = html.Div([
    app_header,
    html.Div(id="body",children=[
    html.H1('This is our Help page'),
    html.Div('This is our help page content.')],style={"paddingTop":header_height+15})
])


#icon credits, put in somewhere near the bottom:
# #<a href="https://www.flaticon.com/free-icons/home" title="home icons">Home icons created by Dave Gandy - Flaticon</a>
#<a href="https://www.flaticon.com/free-icons/question" title="question icons">Question icons created by Dave Gandy - Flaticon</a>
#<a href="https://www.flaticon.com/free-icons/github" title="github icons">Github icons created by Dave Gandy - Flaticon</a>