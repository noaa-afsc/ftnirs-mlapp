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

#if there isn't a specific webapp version available (not provided in .env or by docker build (.dynenv),
#use the value from the git repo.
if "WEBAPP_RELEASE" not in os.environ:
    od = os.getcwd()
    sd = os.path.dirname(os.path.realpath(__file__))
    os.chdir(sd)
    os.chdir("..")
    os.environ["WEBAPP_RELEASE"] = os.popen(f'git describe --tags {os.popen("git rev-list --tags --max-count=1 --first-parent").read()}').read()
    os.chdir(od)

app = Dash(__name__, requests_pathname_prefix=f"/{os.getenv('APPNAME')}/", routes_pathname_prefix=f"/{os.getenv('APPNAME')}/",
                    external_stylesheets=external_stylesheets,
                    use_pages=True)

if __name__ == '__main__':

    app.run(debug=False,host=os.getenv("HOSTIP"),port=int(os.getenv("APPPORT")))

