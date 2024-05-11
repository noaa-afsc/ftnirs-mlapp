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
                    style={'textAlign': 'center', 'marginTop': 20}),
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
                    style={'textAlign': 'center','marginTop': 20}),
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
                    style={'textAlign': 'center','marginTop': 20}),
                html.Hr(style={'marginBottom': 40}),
                html.Div(id = "params-holder"),
            ],style ={"display": "inline-block",'vertical-align': 'top','textAlign': 'right'}),
        ]),

    html.Div(
        [
            html.Hr(style = {'marginTop': 50}),
            html.Button('RUN',id="run-button", style={'textAlign': 'left','vertical-align': 'top','marginLeft': 650}),
            html.Hr()
        ]),
    html.Div(id = "outputs-holder")
])

#refactor to have a single 'parameter holder' as input
@app.callback(
Output('outputs-holder', 'children'),
     Input('run-button', 'n_clicks'),
     State('toggle-mode', 'value'),
     State('model-select', 'value'),
     State('checklist-params', 'value'),
     State('slider-1-params-name', 'children'),
     State('slider-1-params', 'value')
 )
def run_model(n_clicks,mode,model,checklist_params,slider_1_params_name,slider_1_params):

    params = list(zip([slider_1_params_name],[slider_1_params])) #add to internal lists as use more params inputs

    print(params)

    config_out = "Configuration<br>Run mode:"+ "Inference" if mode else "Training" +"\nRun model: "+model+\
                                                 "\nChecklist parameters selected: "+"\n\t-".join(checklist_params)+\
                                                 "\nParameters values selected: "+\
                                                 "".join(["\n\t"+a+":"+str(b) for (a,b) in params])

    return dcc.Textarea(id='config_report',value = config_out,style={'width': 3000,'height': 3000})

@app.callback(#Output('params-select', 'options'),
                    #Output('params-select', 'value'),
                    Output('params-holder', 'children'),
                    Input('toggle-mode', 'value'),
                    Input('model-select', 'value')
)
def get_parameters(mode,model):

    if model is not None:
        #inference
        if mode:
            #we will ideally want this to read properties of the model object itself to run this.
            #hopefully will not be particular to a certain model.

            if model == "hello4.hd5":
                return [dcc.Checklist(id='checklist-params', options=["on_GPU","specific other thing","secret third thing"])]
            else:
                return [dcc.Checklist(id='checklist-params', options=["on_GPU"], value=[])]
        #training
        else:
            #this will largely be populated manually
            if model == "this architecture":
                #return ["test"],[],dcc.Slider(0, 20, 5,value=10,id='test-conditional-component')
                return [dcc.Checklist(id='checklist-params', options=["test"], value=[]),
                        html.Div(id='slider-1-params-name',style={'textAlign': 'left'},children="var1"),
                        dcc.Slider(0, 20, 5,value=10,id='slider-1-params')]
            elif model == "that architecture":
                #return ["on_GPU","other thing"],[],[]
                return [dcc.Checklist(id='checklist-params', options=["on_GPU","other thing"], value=[])]
            elif model == "other thing":
                #return ["on_GPU","other thing","secret third thing"],[],[]
                return [dcc.Checklist(id='checklist-params', options=["on_GPU","other thing","secret third thing"], value=[])]
    else:
        return []


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
