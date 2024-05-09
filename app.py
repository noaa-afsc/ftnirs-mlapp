import os
from dotenv import load_dotenv
import dash
import dash_daq as daq
import plotly.graph_objects as go
from dash import dcc,html,State
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from google.cloud import storage
import base64

external_stylesheets = [dbc.themes.BOOTSTRAP,'https://codepen.io/chriddyp/pen/bWLwgP.css']

if os.getenv("HOSTIP") == None and os.getenv("APPPORT") == None:
    print("using app ip and port defaults")
    # code.interact(local=dict(globals(), **locals()))
    load_dotenv('./tmp/.env')
STORAGE_CLIENT = storage.Client(project=os.getenv("GCP_PROJ"))
DATA_BUCKET = os.getenv("DATA_BUCKET")
TMP_BUCKET = os.getenv("TMP_BUCKET")

def get_objs():
    objs = list(STORAGE_CLIENT.list_blobs(DATA_BUCKET))
    return [i._properties["name"] for i in objs]

def get_datasets():
    return [i[9:] for i in get_objs() if "datasets/" == i[:9]]

def get_models():
    return [i[7:] for i in get_objs() if "models/" == i[:7]]
def get_archetectures():
    return ["this architecture","that architecture","other thing"]

app = dash.Dash(__name__,requests_pathname_prefix ="/ftnirs_mlapp/",routes_pathname_prefix="/ftnirs_mlapp/" ,external_stylesheets=external_stylesheets)

app.layout = html.Div(id='parent', children=[

    html.Div(
        [
            dbc.Alert(
                f"Dataset successfully uploaded!",
                id="alert-dataset-success",
                is_open=False,
                color="success",
                duration=4000),
            dbc.Alert(
                "Dataset upload unsuccessful: did not pass standards checking",
                id="alert-dataset-fail",
                is_open=False,
                color="danger",
                duration=4000),
            dbc.Alert(
                "Model successfully uploaded!",
                id="alert-model-success",
                is_open=False,
                color="success",
                duration=4000),
            dbc.Alert(
                "Model upload unsuccessful: did not pass standards checking",
                id="alert-model-fail",
                is_open=False,
                color="danger",
                duration=4000)
        ]),
    html.Div(
        [
        html.Div(
    [
                html.H2(id='H2_1', children='Select Datasets',
                    style={'textAlign': 'left', 'marginTop': 20}),
                html.Hr(style={'marginBottom': 40}),
                dcc.Checklist(id='dataset-select',
                    options=get_datasets(), # [9:]if f"datasets/" == i[:8]
                    value=[]),
                dcc.Upload(
                    id='upload-ds',
                    children=html.Button('Upload File(s)'),
                    multiple=True)
            ],style={"display": "inline-block",'vertical-align': 'top','textAlign': 'center','marginRight': 200}),

        html.Div(
    [
                html.H2(id='H2_2', children='Select Mode',
                    style={'textAlign': 'left','marginTop': 20}),
                html.Hr(style={'marginBottom': 40}),
                html.Div(
                    [
                        html.Div(id='toggle-mode-output',style = {"display": "inline-block",'textAlign': 'left','width': 200}),
                        daq.ToggleSwitch(id='toggle-mode',
                            value=False,
                            style ={"display": "inline-block"}),
                    ]),
                dcc.Dropdown(id='model-select'),
                dcc.Upload(id='upload-model',children=None,multiple=True)
            ],style ={"display": "inline-block",'vertical-align': 'top','textAlign': 'center','marginRight': 200}),

        html.Div(
            [
                html.H2(id='H2_3', children='Select Parameters',
                    style={'textAlign': 'left','marginTop': 20}),
                html.Hr(style={'marginBottom': 40})
            ],style ={"display": "inline-block",'vertical-align': 'top','textAlign': 'right'}),
        ]),

    html.Div(
        [
            html.Hr(style = {'marginTop': 100,"width": "200%"}),
            html.Button('RUN',id="run-button", style={'textAlign': 'left','vertical-align': 'top','marginLeft': 650}),
            html.Hr()
        ]),
])

@app.callback(
    Output('model-select', 'options'),
    Output('upload-model', 'children'),
    Input('toggle-mode', 'value'),
    Input('alert-model-success', 'is_open')
)
def update_model_checklist(inference,is_open):

    if inference:
        return get_models(),html.Button('Upload Model(s)')
    else:
        return get_archetectures(),None

@app.callback(
    Output('dataset-select', 'options'),
    Input('alert-dataset-success', 'is_open')
)
def update_data_checklist(is_open):
    return get_datasets()

@app.callback(
    Output('toggle-mode-output', 'children'),
    Input('toggle-mode', 'value')
)
def update_output(value):
    return f'mode selected: {"INFERENCE" if value else "TRAINING"}'

@app.callback(
    Output("alert-dataset-success", "is_open"),
    Output("alert-dataset-fail", "is_open"),
    Input('upload-ds', 'filename'),Input('upload-ds', 'contents')
)
def datasets_to_gcp(filename,contents):
    success = True
    #this gets called by alert on startup as this is an output, make sure to not trigger alert in this case
    if not filename == None and not contents == None:
        for i in zip(filename,contents):
            if data_check_datasets(i):

                content_type, content_string = i[1].split(',')
                decoded = base64.b64decode(content_string)
                blob = STORAGE_CLIENT.bucket(DATA_BUCKET).blob(f'datasets/{i[0]}')

                blob.upload_from_string(decoded, 'text/csv')
            else:
                success = False

        return success,not success
    else:
        return False,False

@app.callback(
    Output("alert-model-success", "is_open"),
    Output("alert-model-fail", "is_open"),
    Input('upload-model', 'filename'),Input('upload-model', 'contents')
)
def models_to_gcp(filename, contents):
    success = True
    # this gets called by alert on startup as this is an output, make sure to not trigger alert in this case
    if not filename == None and not contents == None:
        for i in zip(filename, contents):
            if data_check_models(i):

                content_type, content_string = i[1].split(',')
                decoded = base64.b64decode(content_string)
                blob = STORAGE_CLIENT.bucket(DATA_BUCKET).blob(f'models/{i[0]}')

                blob.upload_from_string(decoded, 'text/csv')
            else:
                success = False

        return success, not success
    else:
        return False, False
def data_check_datasets(data):

    if not ".csv" == data[0][-4:]:
        return False

    return True

def data_check_models(data):

    if not ".hd5" == data[0][-4:]:
        return False

    return True


if __name__ == '__main__':

    app.run(debug=False,host=os.getenv("HOSTIP"),port=int(os.getenv("APPPORT")))
