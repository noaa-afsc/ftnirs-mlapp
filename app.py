import dash
import dash_bootstrap_components as dbc
import os
from dotenv import load_dotenv

external_stylesheets = [dbc.themes.BOOTSTRAP,'https://codepen.io/chriddyp/pen/bWLwgP.css']

if os.getenv("HOSTIP") == None and os.getenv("APPPORT") == None:
    print("using app ip and port defaults")
    # code.interact(local=dict(globals(), **locals()))
    load_dotenv('./tmp/.env')

app = dash.Dash(__name__, requests_pathname_prefix="/ftnirs_mlapp/", routes_pathname_prefix="/ftnirs_mlapp/",
                    external_stylesheets=external_stylesheets, use_pages=True)

if __name__ == '__main__':

    app.run(debug=False,host=os.getenv("HOSTIP"),port=int(os.getenv("APPPORT")))

