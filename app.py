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

PARAMS_DICT = {} #global state of currently selected params.
PARAMS_DICT_RUN = {} #global state of last run params.

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
                duration=4000),
            dbc.Alert(
                "Run Failed: no datasets specified",
                id="alert-model-run-ds-fail",
                is_open=False,
                color="danger",
                duration=4000),
            dbc.Alert(
                "Run Failed: no model specified",
                id="alert-model-run-models-fail",
                is_open=False,
                color="danger",
                duration=4000),
            dbc.Alert(
                "Run Failed: error while processing algorithm",
                id="alert-model-run-processing-fail",
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
                    value=[],style={'width':630}),
                dcc.Upload(
                    id='upload-ds',
                    children=html.Button('Upload File(s)'),
                    multiple=True)
            ],style={"display": "inline-block",'vertical-align': 'top','textAlign': 'left','marginRight': 200}),

        html.Div(
    [
                html.H2(id='H2_2', children='Select Mode',
                    style={'textAlign': 'center','marginTop': 20}),
                html.Hr(style={'marginBottom': 40}),
                html.Div(
                    [
                        html.Div(id='toggle-mode-output',style = {"display": "inline-block",'textAlign': 'left','width': 200}),
                        daq.ToggleSwitch(id='toggle-mode',
                            value=True,
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
            html.Button('RUN',id="run-button"),
            html.Div(id='run-message',children=[]),
            html.Hr()
        ], style={'textAlign': 'center','vertical-align': 'top'}), #,'marginLeft': 650
html.Div(
        [
        html.Div(
        [
                    html.Div(id='config-report',children =[],style={'textAlign': 'left','vertical-align': 'top','width': 400,'height': 300})#
             ],style={"display": "inline-block",'vertical-align': 'top','textAlign': 'center','marginRight': 200}),
        html.Div(
            [
                    html.Div(id = "artifacts-holder"),
                    html.Div(id = "stats-holder")
            ],style={"display": "inline-block",'vertical-align': 'top','textAlign': 'center','marginRight': 200}),
        html.Div(
                [
                    html.Button("Download Results", id="btn-download-results"),
                    dcc.Download(id="download-results"),
                    html.Div(id = "train-out")
                ],style={"display": "inline-block",'vertical-align': 'top'})
        ],style={"display": "inline-block",'vertical-align': 'top'})

])
@app.callback(
    Output('alert-model-run-processing-fail','is_open'),
     Output('alert-model-run-ds-fail','is_open'),
     Output('alert-model-run-models-fail','is_open'),
     Output('run-message', 'children'),
     Output('config-report', 'children'),
     Output('train-out', 'children'),
     #Output('output-data', 'value'),
     #Output('artifacts', 'value'),
     #Output('artifacts', 'value'),
     #Output('stats', 'value'),
     Input('run-button', 'n_clicks'),
     State('toggle-mode', 'value'),
     State('model-select', 'value'),
     State('dataset-select', 'value')
 )
def run_model(n_clicks,mode,model,datasets):

    if n_clicks is not None:

        global PARAMS_DICT_RUN

        PARAMS_DICT_RUN = PARAMS_DICT

        message = "Run Failed: "

        processing_fail = False
        ds_fail = False
        model_fail = False

        any_error = False

        if len(datasets)==0:

            message = message + "no datasets specified"
            ds_fail = True
            config_out_payload = []

            any_error = True

            #payload = False,True,False,"Run Failed: ",[]
        if model is None:
            if any_error:
                message = message + " & no model specified"
            else:
                message = message + "no model specified"
            model_fail = True

            any_error = True
            config_out_payload = []
            #payload = False,False,False,"Run Failed: no model specified",[]


        if not any_error:
            try:

                #config_out = "Run Configuration:\n +\"Mode:"+ ("Inference" if mode else "Training")
                #                                            "\nModel: "+model+\
                #                                            "\nDatasets: " + "".join(["\n\t-" + str(i) for i in datasets])+\
                #                                            "\nParameters selected: "+ "".join(["\n\t-"+a+":"+str(b) for (a,b) in PARAMS_DICT.items()])

                config_out_children = [html.Div(id='run-name-block',children =[html.Div(id='run-name-prompt',children = "Run name:"),
                                            dcc.Input(id='run-name', type="text", placeholder="my unique run name",style={'textAlign': 'left', 'vertical-align': 'top', 'width': 400})
                                                                                ],style={"display": "inline-block"}) if not mode else ""] +\
                                       [html.Div(id='config-report-rc',children = "Run Configuration:"),
                                       html.Div(id='config-report-mode', children="Mode: "+ ("Inference" if mode else "Training")),
                                       html.Div(id='config-report-model',children="Model: " + model),
                                       html.Div(id='config-report-datasets',children ='Datasets: ')] +\
                                       [html.Div(id='config-report-datasets-' + i,children="- "+str(i),style={'marginLeft': 15}) for i in datasets] +\
                                       [html.Div(id='config-report-parameters',children ='Parameters: ')] + \
                                       [html.Div(id='config-report-parameters-' + a,children="- "+a+": "+str(b),style={'marginLeft': 15}) for (a,b) in PARAMS_DICT.items()]
                #this is where model will actually run
                #data,artifacts,stats = run_model(mode,model,datasets)

                #html.Div(id='config-report-mode'),
                #html.Div(id='config-report-model'),
                #html.Div(id='config-report-datasets'),
                #html.Div(id='config-report-parameters')]

                message = "Run Succeeded!"

                config_out_payload = config_out_children
            except Exception as e:
                message = "Run Failed: error while processing algorithm"
                config_out_payload = [html.Div(id='error title',children="ERROR:"),html.Div(id='error message',children =[str(e)])]
                processing_fail = True

        return [processing_fail,ds_fail,model_fail,message,config_out_payload,html.Button("Upload Trained model", id="btn-upload-model") if not mode else ""]

@app.callback(#Output('params-select', 'options'),
                    #Output('params-select', 'value'),
                    Output('params-holder', 'children'),
                    Input('toggle-mode', 'value'),
                    Input('model-select', 'value')
)
def get_parameters(mode,model):

    global PARAMS_DICT

    PARAMS_DICT = {}

    if model is not None:
        #inference
        if mode:
            #we will ideally want this to read properties of the model object itself to run this.
            #hopefully will not be particular to a certain model.

            if model == "hello4.hd5":
                return [dcc.Checklist(id='checklist-params', options=["on_GPU","specific other thing","secret third thing"], value=[False,False,False])]
            else:
                return [dcc.Checklist(id='checklist-params', options=["on_GPU"], value=[False])]
        #training
        else:
            #this will largely be populated manually (hardcoded).  Right now, that will look like Michael explaining to Dan
            #the relevant training hyperparameters to expose per modeling approach ("archetecture").
            if model == "this architecture":
                #return ["test"],[],dcc.Slider(0, 20, 5,value=10,id='test-conditional-component')
                return [dcc.Checklist(id='checklist-params', options=["test"], value=[]),
                        html.Div(id='var1-param-name',style={'textAlign': 'left'},children="var1"),
                        dcc.Slider(0, 20, 5,value=10,id='var1-param')]
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
        return get_models(),html.Button('Upload Trained Model(s)')
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

################################as add parameters, give it this boilerplate:

@app.callback(Input('checklist-params', 'value'),
              Input('checklist-params', 'options'),
              #Input('run-button', 'n_clicks'),
              Input('params-holder', 'children')
)
def checklist_params_dict_pop(value,options,_):

    global PARAMS_DICT

    out_dict = {i: (True if i in value else False) for i in options}
    for i in out_dict:
        PARAMS_DICT[i]= out_dict[i]

    print(PARAMS_DICT)
@app.callback(Input('var1-param', 'value'),
              Input('params-holder', 'children')
)
def checklist_params_dict_pop(value,_):
    global PARAMS_DICT

    PARAMS_DICT["var1"] = value

    print(PARAMS_DICT)




if __name__ == '__main__':

    app.run(debug=False,host=os.getenv("HOSTIP"),port=int(os.getenv("APPPORT")))
