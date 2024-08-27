from dash import Dash
import dash_bootstrap_components as dbc
import os
from dotenv import load_dotenv

external_stylesheets = [dbc.themes.BOOTSTRAP,'https://codepen.io/chriddyp/pen/bWLwgP.css']

#check for any env files. Local uses ./tmp/.env, and cloud build uses both docker env (static) and ./tmp

if os.path.isfile('./tmp/.env'):
    load_dotenv('./tmp/.env')

if os.path.isfile('./tmp/.dynenv'):
    load_dotenv('./tmp/.dynenv')

app = Dash(__name__, requests_pathname_prefix=f"/{os.getenv('APPNAME')}/", routes_pathname_prefix=f"/{os.getenv('APPNAME')}/",
                    external_stylesheets=external_stylesheets,
                    use_pages=True)

if __name__ == '__main__':

    app.run(debug=False,host=os.getenv("HOSTIP"),port=int(os.getenv("APPPORT")))

