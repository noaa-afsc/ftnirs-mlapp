from dash import Dash,Output,Input,DiskcacheManager
import dash_bootstrap_components as dbc
import os
from dotenv import load_dotenv
from ftnirsml.constants import TRAINING_APPROACHES
import diskcache

external_stylesheets = [dbc.themes.BOOTSTRAP,'https://codepen.io/chriddyp/pen/bWLwgP.css']

#check for any env files. Local uses ./tmp/.env, and cloud build uses both docker compose env (static) and ./tmp

if os.path.isfile('../tmp/.env'):
    load_dotenv('../tmp/.env')

if os.path.isfile('../tmp/.dynenv'):
    load_dotenv('../tmp/.dynenv')

#if there isn't a specific webapp version available (not provided in .env or by docker build (.dynenv),
#use the value from the git repo.
if "WEBAPP_RELEASE" not in os.environ:
    od = os.getcwd()
    sd = os.path.dirname(os.path.realpath(__file__))
    os.chdir(sd)
    os.chdir("..")
    os.environ["WEBAPP_RELEASE"] = os.popen(f'git describe --tags {os.popen("git rev-list --tags --max-count=1 --first-parent").read()}').read()
    os.chdir(od)

CACHE = diskcache.Cache("./cache2")
background_callback_manager  = DiskcacheManager(CACHE)

app = Dash(__name__, requests_pathname_prefix=f"/{os.getenv('APPNAME')}/", routes_pathname_prefix=f"/{os.getenv('APPNAME')}/",
                    external_stylesheets=external_stylesheets,
                    use_pages=True, update_title=None,background_callback_manager =background_callback_manager)

#attempt to add some dynamic callbacks in here, since with app.callback (with app object available) I can use
#lambda decorators to deal with bool variables.
bool_callbacks_needed = {}
for i in TRAINING_APPROACHES.keys():
    if 'parameters' in TRAINING_APPROACHES[i]:
        for m in TRAINING_APPROACHES[i]['parameters'].keys():
            if TRAINING_APPROACHES[i]['parameters'][m]["data_type2"]==bool:
                bool_callbacks_needed[m]=TRAINING_APPROACHES[i]['parameters'][m]

#this is ill as hell!
for i in bool_callbacks_needed:
    app.callback(Output(i + "-title", "children"),Input(i, 'value'))(lambda val: f"{bool_callbacks_needed[i]['display_name'].split(':')[0]}: {val}")

if __name__ == '__main__':

    app.run(debug=False,host=os.getenv("HOSTIP"),port=int(os.getenv("APPPORT")))

