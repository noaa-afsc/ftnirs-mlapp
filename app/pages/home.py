import csv
from collections import Counter
import os
import tarfile
import io
import ast
import dash
import dash_daq as daq
from dash import dcc,html, dash_table, callback,  Output, Input, State
import dash_bootstrap_components as dbc
from datetime import datetime,timedelta
import pandas as pd
import random as rand
from google.cloud import storage
import plotly.graph_objects as go
import time
import base64
import h5py
#import app_data
from ftnirsml.constants import WN_MATCH,INFORMATIONAL,RESPONSE_COLUMNS,SPLITNAME,WN_STRING_NAME,IDNAME,STANDARD_COLUMN_NAMES,MISSING_DATA_VALUE,ONE_HOT_FLAG,MISSING_DATA_VALUE_UNSCALED,TRAINING_APPROACHES
import diskcache
#safer and less bespoke than what I previously implemented.
import uuid
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv
from app_constant import app_header,header_height,encode_image
from ftnirsml.code import loadModelWithMetadata, wnExtract, format_data, TrainingModeWithoutHyperband, TrainingModeWithHyperband,  hash_dataset, InferenceMode, TrainingModeFinetuning
#import zipfile
import tempfile
import logging
from tensorflow.keras.callbacks import EarlyStopping, Callback as tk_callbacks
class EpochCounterCallback(tk_callbacks):

    def __init__(self,session_id,logger,run_id):
        super().__init__()
        self.session_id = session_id
        self.logger = logger
        self.run_id = run_id
    def on_epoch_begin(self, epoch, logs=None):
        self.logger.debug(f"{self.session_id} Current Epoch: {epoch+1} (rid: {self.run_id[:6]}...)")

load_dotenv('../tmp/.env')

dash.register_page(__name__, path='/',name = os.getenv("APPNAME")) #

#declare diskcache, to manage global variables/long running jobs in a way that
#is multiple user and thread safe and fault tolerant.

CACHE = diskcache.Cache("./cache")
CACHE_EXPIRY = 1 #in days

#set up info level manually triggered logger
log_handler = RotatingFileHandler("session_logs.log",maxBytes=1*1024*1024)
log_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_handler.setFormatter(formatter)

LOGGER_MANUAL = logging.getLogger('manual_logger')
LOGGER_MANUAL.setLevel(logging.DEBUG)
LOGGER_MANUAL.addHandler(log_handler)

#this should be an .env variable
GCP_PROJ="ggn-nmfs-afscftnirs-dev-97fc"

#STORAGE_CLIENT = storage.Client(project=os.getenv("GCP_PROJ"))
STORAGE_CLIENT = storage.Client(project=GCP_PROJ)
DATA_BUCKET = os.getenv("DATA_BUCKET")
TMP_BUCKET = os.getenv("TMP_BUCKET")

PRETRAINED_MODEL_METADATA_PARAMS = ["wave_number_min", "wave_number_max", "wave_number_step", "wn_filter", "wn_scaler",
                                  "bio_scaler", "response_scaler"]



def get_objs():
    objs = list(STORAGE_CLIENT.list_blobs(DATA_BUCKET))
    return [i._properties["name"] for i in objs]

#should have this return elegible datasets as 1st tuple inelibile as 2nd tuple. (requires metadata check every call... )?
def get_datasets():
    datasets = [i for i in get_objs() if "datasets/" == i[:9]]

    return [{"value": x[9:], "label": f"{x[9:]} - nrow: {STORAGE_CLIENT.bucket(DATA_BUCKET).get_blob(x).metadata['data_rows']}", "disabled": False} for x in datasets]

def get_pretrained():
    return [i[7:] for i in get_objs() if "models/" == i[:7]]

#some graphical variables:
top_row_max_height = 700
horizontal_pane_margin = 20#50
left_col_width = "30%" #500
middle_col_width = "30%" #400
right_col_width = "30%" #400
H2_height = 50
H2_height_below_padding = 30
checklist_pixel_padding_between = "5px"
left_body_width ="95%"

refresh_button_width = 40

#this will be supplied in callbacks to make the loading icon show up on button click
ref_content = [html.Img(src=encode_image("./static/refresh-arrow.png"),style={"width":"20px","height":'auto'})]

#sort the params in TRAINING_APPROACHES, then add the dash stuff to the existing dict.
#based on the number of bool variables, spin off some dynamic callback functions as needed.
#        TRAINING_APPROACHES[i]['dash_parameters']=html.Div([]
for i in TRAINING_APPROACHES.keys():
    if 'parameters' in TRAINING_APPROACHES[i]:
        obj = []
        for m in TRAINING_APPROACHES[i]["parameters"]:
            if TRAINING_APPROACHES[i]["parameters"][m]["data_type2"]==bool:
                obj.append(html.Div(id=m+'-title',children=TRAINING_APPROACHES[i]["parameters"][m]["display_name"]))
                obj.append(daq.ToggleSwitch(id=m, value=TRAINING_APPROACHES[i]["parameters"][m]["default_value"]))
            else:
                obj.append(html.Div(TRAINING_APPROACHES[i]["parameters"][m]["display_name"]))
                obj.append(dcc.Input(id=m, type=TRAINING_APPROACHES[i]["parameters"][m]["data_type"], \
                                                         placeholder=str(TRAINING_APPROACHES[i]["parameters"][m]["data_type2"]).split(" ")[1][1:-2] + \
                                     " (default: " + str(TRAINING_APPROACHES[i]["parameters"][m]["default_value"])+")"))
        print(f"added to training approaches for {i}")
        TRAINING_APPROACHES[i]["dash_params"]=html.Div(obj)

APP_STORAGE_TYPE = 'memory' #memory, session, local

WN_FILTER_DEFAULT,WN_SCALING_DEFAULT,BIO_SCALING_DEFAULT,RESPONSE_SCALING_DEFAULT = 'savgol','MinMaxScaler','MinMaxScaler','MinMaxScaler'

interp_children_choose = [html.Div(id='interp_status',children='Custom interpolation: False'),
                    daq.ToggleSwitch(id='define-interp',value=False,style={"width":"50%"}),
                    html.Div(id="interp-choice")]

layout = html.Div(id='parent', children=[
    app_header,
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
                duration=4000),
            dbc.Alert(
                "Trained model successfully uploaded from session",
                id="model-upload-from-session-success",
                is_open=False,
                color="success",
                duration=4000),
            dbc.Alert(
                "Trained model upload failure: please specify a unique model name",
                id="model-upload-from-session-failure",
                is_open=False,
                color="danger",
                duration=4000)
        ],style={"position":"fixed","top":header_height+15,"zIndex":999,'width':"100%"}),
    html.Div(id = 'toprow',children=[
        dcc.Interval(id='interval-component',interval=1*1000, n_intervals=0),
        dcc.Store(id='session-id', storage_type='session', data="session_init"), #see if this can be recoverable, potentially. s
        dcc.Store(id='params_dict', storage_type=APP_STORAGE_TYPE,data = {}),
        dcc.Store(id='dataset_titles', storage_type=APP_STORAGE_TYPE,data = {}),
        dcc.Store(id='data_metadata_dict', storage_type=APP_STORAGE_TYPE,data = {}),
        dcc.Store(id='columns_dict', storage_type=APP_STORAGE_TYPE, data={"wav":[WN_STRING_NAME],'std':[i for i in STANDARD_COLUMN_NAMES],'oc':[]}),
        #dcc.Store(id='columns_dict', storage_type='memory', data={"wav":{WN_STRING_NAME},'std':{i for i in STANDARD_COLUMN_NAMES},'oc':set()}), #more effecient but doesn't work with framework
        dcc.Store(id='pretrained_model_metadata_dict', storage_type=APP_STORAGE_TYPE, data={}),
        dcc.Store(id='pretrained_value_dict', storage_type=APP_STORAGE_TYPE, data={'value':None}),
        dcc.Store(id='approaches_value_dict', storage_type=APP_STORAGE_TYPE, data={'value': None}),
        #dcc.Store(id='refresh-ds-count', storage_type='memory', data = 0),
        dcc.Store(id='run_id', storage_type=APP_STORAGE_TYPE),
        dcc.Store(id='dataset_select_last_clicked', storage_type=APP_STORAGE_TYPE,data = False),
            html.Div(id='left col top row',children=[
            html.Div(id='datasets',children=[
                html.H2(id='H2_1', children='Select Datasets',
                    style={'textAlign': 'center','marginBelow':H2_height_below_padding, 'height':H2_height}),  #'marginTop': 20}
                #todo: maybe I can use the similar syntax like in data pane to add 'n' after the title- important info for using datasets. Or, make a seperate page for data and
                #todo signal explorer.
                dcc.Checklist(id='dataset-select',
                    options=get_datasets(), # [9:]if f"datasets/" == i[:8]
                    value=[], style={'maxHeight':200,'overflowY':'auto','width':left_body_width},inputStyle={"margin-right":checklist_pixel_padding_between }),
                html.Div(id='ds-buttons',children=[
                    html.Div([html.Button(id='refresh-ds',children=[dcc.Loading(html.Div(id='ds_ref_content',children=ref_content),type="circle")],style={'width':refresh_button_width,"align-items":'center','padding':0})],style={"display": "inline-block"}), #,'margin': 0
                    html.Div([dcc.Upload(
                        id='upload-ds',
                        children=html.Button('Upload Dataset(s)'),
                        multiple=True)],style={"display": "inline-block"})
                ]),
            ],style={'vertical-align': 'top','textAlign': 'left'}), #'marginRight': 20 #"display": "inline-block",

            html.Div(id='modes and models',children=
                [
                    html.H2(id='H2_2', children='Select Mode',
                            style={'textAlign': 'center', 'marginTop': 20, 'height':H2_height,'marginBelow':H2_height_below_padding}),
                    html.Div(id="modes_and_models_body",children=[
                        dcc.Dropdown(id='mode-select', value="Training", clearable=False,
                                     options=["Training", "Inference", "Fine-tuning"],style={'width': 200}), #
                        html.Div(id='mode-select-output', style={'textAlign': 'left'}),
                        html.Div(id = "approaches-holder"),
                        html.Div(id = "pretrained-holder")], style={'vertical-align': 'top', 'textAlign': 'center','height': 375,'maxHeight': 375,"overflowY":'auto','width':left_body_width})])
                ],style={"display": "inline-block",'marginRight': horizontal_pane_margin,'height': top_row_max_height, 'width': left_col_width}),
        #,style={"display": "inline-block",'marginRight': horizontal_pane_margin,'height': top_row_max_height, 'width': left_col_width}),

        html.Div(
            [
                html.H2(children='Data Columns',
                        style={'textAlign': 'center','height':H2_height,'marginBelow':H2_height_below_padding}),
                html.Div(id="data-pane",style={'height':top_row_max_height-H2_height-5,'maxHeight':top_row_max_height-H2_height-5,'overflowY':'auto'})
            ], style={"display": "inline-block", 'vertical-align': 'top', 'textAlign': 'left','marginRight': horizontal_pane_margin,'width': middle_col_width,'height':top_row_max_height,'maxHeight':top_row_max_height}), #,

        html.Div(
            [
                html.H2(id='H2_3', children='Select Parameters',
                    style={'textAlign': 'center','marginBelow':H2_height_below_padding, 'height':H2_height}), #,'marginTop': 20
                html.Div(id = "params-holder",children=[
                    html.Div(id="preprocessing_params",children=[
                    html.Div(WN_STRING_NAME+" filter choice:"),
                    dcc.Dropdown(id='wn-filter',options=['savgol','moving_average','gaussian','median','wavelet','fourier','pca'], value=WN_FILTER_DEFAULT),
                    html.Div(WN_STRING_NAME+" scaling choice:"),
                    dcc.Dropdown(id='wn-scaling',options=['MinMaxScaler', 'StandardScaler', 'MaxAbsScaler', 'RobustScaler', 'Normalizer'], value=WN_SCALING_DEFAULT),
                    html.Div("Biological scaling choice:"),
                    dcc.Dropdown(id='bio-scaling',options=['MinMaxScaler', 'StandardScaler', 'MaxAbsScaler', 'RobustScaler', 'Normalizer'], value=BIO_SCALING_DEFAULT),
                    html.Div("Response scaling choice:"),
                    dcc.Dropdown(id='response-scaling',options=['MinMaxScaler', 'StandardScaler', 'MaxAbsScaler', 'RobustScaler'], value=RESPONSE_SCALING_DEFAULT),
                    html.Div(id='interp_holder',children=interp_children_choose)],style={"width": "50%"}),
                    html.Div(id='mode_params',children=[]),
                    html.Div(id="approach_params")
                ]),
            ],style ={"display": "inline-block",'vertical-align': 'top','textAlign': 'left','width':right_col_width,'height':top_row_max_height,'overflowY':'auto'}), #'maxHeight':top_row_max_height
        ],style={'height':top_row_max_height,"paddingTop":header_height}),html.Hr(), #style={'marginBottom': 60}
    html.Div(id='middle_row',children=[
        dcc.Markdown(id='manual-log-holder',style={'textAlign': 'left','vertical-align': 'top','width': left_col_width,'maxHeight':100,'height': 100,"display": "inline-block",'overflowY':'auto'}),
        html.Div(
            [
                html.Button('RUN',id="run-button",style={"font-weight": 900,"width":100,"height":100}),
                dcc.Loading(id='run-message',children=[])
            ], style={'textAlign': 'center','vertical-align': 'top',"width":middle_col_width,"display": "inline-block"}),
        dcc.Markdown(id='redirect-log-holder',style={'textAlign': 'left','vertical-align': 'top','width': left_col_width,'maxHeight':100,'height': 100,"display": "inline-block",'overflowY':'auto'}),
        ],style={"display": "inline-block","width":"100%"}),html.Hr(style={'marginBottom': 40}), #,'marginLeft': 650
html.Div(
        [
        html.Div(
        [
                    html.Div(id='config-report',children =[],style={'textAlign': 'left','vertical-align': 'top','width': left_body_width,'height': 300})#
             ],style={"display": "inline-block",'vertical-align': 'top','textAlign': 'center','marginRight': horizontal_pane_margin,'width': left_col_width}),
        html.Div(
            [
                    html.Div(id="stats-out"),
                    html.Div(id = "artifacts-out")
            ],style={"display": "inline-block",'vertical-align': 'top','textAlign': 'center','marginRight': horizontal_pane_margin,'width': middle_col_width, 'marginRight': horizontal_pane_margin}),
        html.Div(
                [

                    html.Div(id = "download-out"),
                    html.Div(id = "upload-out")
                ],style={"display": "inline-block",'vertical-align': 'top','width':right_col_width})
        ],style={'vertical-align': 'top'}) #"display": "inline-block",

])

@callback(Output('manual-log-holder',"children"),
          Output('redirect-log-holder', "children"),
          Output("session-id", "data"),
          Input("interval-component",'n_intervals'),
          State("session-id", "data"))
def update_logs(n_intervals,data):

    #assign a new session a uuid4
    if data=="session_init":
        data = str(uuid.uuid4())

    #read the log entries from the last second and populate
    with open('session_logs.log','r') as log_file:

        #import code
        #code.interact(local=dict(globals(), **locals()))
        manual = [line[:19]+" "+line[70:] for line in log_file if (data in line and line[26:30] == 'INFO' and abs((datetime.strptime(line[0:19],"%Y-%m-%d %H:%M:%S")-datetime.now())) < timedelta(days=1))]

    if len(manual) > 0:
        manual = manual[-100:]
        manual = list(reversed(manual))
        manual = "\>"+"\n\>".join(manual) +"\n"

    with (open('session_logs.log', 'r') as log_file):
        #import code
        #code.interact(local=dict(globals(), **locals()))
        redirect =[line[:19]+" "+line[70:] for line in log_file if (data in line and line[26:31] == 'DEBUG' and abs((datetime.strptime(line[0:19],"%Y-%m-%d %H:%M:%S")-datetime.now())) < timedelta(days=1))]


    if len(redirect)>0:
        redirect = redirect[-100:]
        redirect = list(reversed(redirect))
        redirect = "\>" + "\n\>".join(redirect) + "\n"

    return manual,redirect,data

@callback(
    Output('approaches_value_dict', 'data'),
    Input('approaches-select', 'value'),
    prevent_initial_call = True
)
def update_approaches_value(approaches_value):

    return {"value":approaches_value}

@callback(
    Output('pretrained_value_dict', 'data'),
    Input('pretrained-select', 'value'),
    prevent_initial_call = True
)
def update_pretrained_value(pretrained_value):

    return {"value":pretrained_value}

@callback(
    Output('pretrained-select', 'options'),
    Output('pm_ref_content', 'children'),
    Input('refresh-pm', 'n_clicks'),
)
def update_pretrained_select(clicks):

    return get_pretrained(),ref_content

@callback(
    Output('pretrained-holder', 'children', allow_duplicate=True),
    Input('mode-select', 'value'),
    Input('alert-model-success', 'is_open'),
    State('pretrained_value_dict', 'data'),
    prevent_initial_call = True
)
def update_pretrained_checklist(mode,is_open,pretrained_value):

    if mode != "Training":
        return [html.H4("Pretrained models:", style={'textAlign': "left"}),
dcc.Dropdown(id='pretrained-select', style={'width': 350}, options=get_pretrained(),value=pretrained_value['value']),
html.Div(id="pretrained-present", style={'textAlign': 'left'}),
        html.Div(id="button-holder-pretrained",children=[
            html.Div([html.Button(id='refresh-pm',children=[dcc.Loading(html.Div(id='pm_ref_content',children=ref_content),type="circle")],style={'width':refresh_button_width,"align-items":'center','padding':0,"marginLeft":0})],style={"display": "inline-block"}),
            html.Div(dcc.Upload(id='upload-pretrained', children=html.Button('Upload Pretrained Model(s)'), multiple=True, style={'textAlign': 'left'}),style={"display": "inline-block"})
    ],style={'textAlign': 'left'})]
    else:
        return []
@callback(
    Output('approaches-holder', 'children'),
    Input('mode-select', 'value'),
    Input('pretrained_model_metadata_dict', 'data'),
    State('pretrained_value_dict', 'data'),
    State('approaches_value_dict', 'data')
)
def update_approach_checklist(mode,pretrain_dict,pretrain_val,approach_val):

    print('in update_approach_checklist')

    pretrain_val = pretrain_val["value"]

    print('dict')
    print(pretrain_dict)
    print('val')
    print(pretrain_val)

    opts = TRAINING_APPROACHES

    if mode != "Inference":

        if mode == "Fine-tuning":

            print('fine tune opts')
            opts = {key: val for (key, val) in opts.items() if opts[key]['finetunable']}

            print(opts)

            #may change this flow depending on whether we include training approaches in model metadata or not
            if pretrain_dict != {}:
                #if/when we add this, make sure we are correctly converting compat_opts value to list
                if pretrain_val != None:
                    if 'compatible_approaches' in pretrain_dict[pretrain_val]:
                        compat_opts = pretrain_dict[pretrain_val]['compatible_approaches']

                        opts = {key:val for (key,val) in opts.items() if key in compat_opts}

            #print(opts)


        return [html.H4("Training Approaches:", style={'textAlign': "left"}),
            dcc.Dropdown(id='approaches-select', style={'width': 200}, options=[i for i in opts],value=approach_val["value"] if approach_val["value"] in opts else None), #[] if approach_val["value"]==None else approach_val["value"]
            html.Div(id="approaches-present", style={'textAlign': 'left'})]
    else:
        return None
@callback(
    Output('approaches-present', 'children'),
    Input("approaches-value_dict", "data"),
    Input("mode-select", "value")
)
def present_approach_metadata(approach,mode):
    approach = approach["value"]

    hidden_metadata_keys = set({"finetunable"})

    if mode != "Inference" and approach != None:

        approach_metadata = TRAINING_APPROACHES[approach]

        obj = [html.Div(children=[html.Strong(str(key)), html.Span(f": {approach_metadata[key]}")], style={'marginBottom': 10}) for key in approach_metadata if key not in hidden_metadata_keys]


        return [html.Div(children=obj)] #html.H5("Approach Info:"),
    else:
        return None

@callback(
    Output('wn-filter', 'value', allow_duplicate=True),
    Output('wn-filter', 'disabled', allow_duplicate=True), #pretrained-present no allow duplicate
    Output('wn-scaling', 'value', allow_duplicate=True),
    Output('wn-scaling', 'disabled', allow_duplicate=True),
    Output('bio-scaling', 'value', allow_duplicate=True),
    Output('bio-scaling', 'disabled', allow_duplicate=True),
    Output('response-scaling', 'value', allow_duplicate=True),
    Output('response-scaling', 'disabled', allow_duplicate=True),
    Output('interp_holder', 'children', allow_duplicate=True),
    Output('pretrained-present', 'children'),
    Input('pretrained_model_metadata_dict', "data"),
    Input("mode-select", "value"),
    State("pretrained_value_dict", "data"),
    prevent_initial_call = True
)
def present_pretrained_metadata(pretrained_model_metadata_dict,mode,pretrained):

    pretrained = pretrained['value']

    if mode != "Training":

        if pretrained!= None:
            #description = pretrained_model_metadata_dict[pretrained].pop("description")

            interp_children_preset = [html.Div(id='interp_status', children='Custom interpolation: True'),
                                      daq.ToggleSwitch(id='define-interp', value=True, disabled=True),
                                      html.Div("Min:"),
                                      dcc.Input(id="interp-min", type="number", value = pretrained_model_metadata_dict[pretrained]['wave_number_min'],disabled=True),
                                      html.Div("Max:"),
                                      dcc.Input(id="interp-max", type="number", value = pretrained_model_metadata_dict[pretrained]['wave_number_max'],disabled=True),
                                      html.Div("Step:"),
                                      dcc.Input(id="interp-step", type="number", value = pretrained_model_metadata_dict[pretrained]['wave_number_step'],disabled=True)]

            #pretrained_metadata = sorted(pretrained_model_metadata_dict[pretrained].items())
            pretrained_metadata = sorted({l:pretrained_model_metadata_dict[pretrained][l] for l in pretrained_model_metadata_dict[pretrained] if l not in PRETRAINED_MODEL_METADATA_PARAMS}.items())

            obj = [html.Div(children=[html.Strong(str(a)), html.Span(f": {b}")],
                            style={'marginBottom': 10}) for a,b in pretrained_metadata] #[l for l in pretrained_metadata if l not in PRETRAINED_MODEL_METADATA_PARAMS]]
            #import code
            #code.interact(local=dict(globals(), **locals()))
            return pretrained_model_metadata_dict[pretrained]["wn_filter"],True,\
                   pretrained_model_metadata_dict[pretrained]["wn_scaler"],True,\
                   pretrained_model_metadata_dict[pretrained]["bio_scaler"],True,\
                   pretrained_model_metadata_dict[pretrained]["response_scaler"],True, \
                   interp_children_preset,obj #html.H5("Pretrained Info:"),  #[html.H4("Pretrained set parameters"),
        else:
            return WN_FILTER_DEFAULT,False,WN_SCALING_DEFAULT,False,BIO_SCALING_DEFAULT,False,RESPONSE_SCALING_DEFAULT,False,interp_children_choose,None

    else:
        return WN_FILTER_DEFAULT,False,WN_SCALING_DEFAULT,False,BIO_SCALING_DEFAULT,False,RESPONSE_SCALING_DEFAULT,False,interp_children_choose,None

@callback(
    Output('data-pane',"children"),
    #Input('approaches-select', "children"),
    Input('data_metadata_dict', "data"),
    Input('pretrained_model_metadata_dict', "data"),
    State('dataset-select', 'value'),
    State('mode-select', 'value'),
    State('pretrained_value_dict', 'data'),
    State('approaches_value_dict', 'data'),
    State('columns_dict', 'data'),
    prevent_initial_call = True
)

def present_columns(data_dict,model_dict,datasets,mode,pretrained_val,approach_val,previous_selections): #(datasets,models,data_dict,model_dict):

    pretrained_val = pretrained_val['value']

    #clear this out to make conditional logic cleaner within
    if mode=='Training':
        pretrained_val = None
    #what do we need to know for columns for model run?
    #training:
    #1. what standard columns and other columns are being used (and their order)
    #inference:
    #1. what standard columns and other columns can be used, as well as what's missing
    #fine tuning:
    #1 and #2 from above


    #flatten lists and count instances of each item. Display this info along with column.
    #add titles for each section.
    #add in logic and indicators for model compatibility.

    #prevent unnecessary remaining logic in fxn and importantly a divide by 0 case
    if len(datasets) == 0:
        return None

    standard_excluded = []
    other_excluded = []

    ds_count = len(datasets)

    wave_counts = []
    #valid waves: increments for every dataset that has valid waves (not = -1, signaling not present).
    valid_waves= 0
    for i in datasets:
        if data_dict[i]['wave_number_end_index']!="['-1']":
            #print(data_dict[i]['wave_number_end_index'])
            wave_counts.append(f"{int(ast.literal_eval(data_dict[i]['wave_number_end_index'])[0])-int(ast.literal_eval(data_dict[i]['wave_number_start_index'])[0])}" + \
                               f";{round(float(ast.literal_eval(data_dict[i]['wave_number_min'])[0]),2)};{round(float(ast.literal_eval(data_dict[i]['wave_number_max'])[0]),2)};" +\
                               f"{round(float(ast.literal_eval(data_dict[i]['wave_number_step'])[0]),2)}")
            valid_waves = valid_waves + 1

    print('wave_counts 1')
    print(wave_counts)
    wave_counts = set(wave_counts)

    print('wave_counts set')
    print(wave_counts)

    #equivalent: how many ds have the same exact same wave_counts.

    #for training and inference, do not allow training for partial presence of wav numbers
    #equivalent = f"{(ds_count-len(wave_counts)+1)}/{ds_count}" if ds_count==0 else "NA"
    wav_str = f"{WN_STRING_NAME} valid: {valid_waves}/{ds_count}, equivalent: {(ds_count-len(wave_counts)+1)}/{ds_count}"

    print(f"mode:{mode} pretrained_val:{pretrained_val}")
    if pretrained_val != None: #and mode is not 'Training':

        pretrained_include = [i.split(ONE_HOT_FLAG)[0] if ONE_HOT_FLAG in i else i for i in ast.literal_eval(model_dict[pretrained_val]['bio_columns'])] + [WN_STRING_NAME]
        pretrained_include_std = set([l for l in pretrained_include if l in STANDARD_COLUMN_NAMES])
        standard_cols_counter = Counter([i for sublist in [ast.literal_eval(data_dict[i]['standard_columns']) for i in datasets] for i in sublist])
        standard_cols_zero_counts = [l for l in pretrained_include_std if l not in standard_cols_counter]
        for l in standard_cols_zero_counts:
            standard_excluded.append((str(l),f'{l} ({0}/{ds_count}) (included in pretrained model)',0))

        pretrained_include_oc = [l for l in pretrained_include if l not in STANDARD_COLUMN_NAMES and l != WN_STRING_NAME]
        other_cols_counter = Counter(
            [i for sublist in [ast.literal_eval(data_dict[i]['other_columns']) for i in datasets] for i in sublist])
        other_cols_counter.update({a: 0 for a in [l for l in pretrained_include_oc if l not in standard_cols_counter]})
        other_cols_zero_counts = [l for l in pretrained_include_oc if l not in other_cols_counter]
        for l in other_cols_zero_counts:
            other_excluded.append((str(l), f'{l} ({0}/{ds_count}) (included in pretrained model)', 0))

    else:
        standard_cols_counter = Counter([i for sublist in [ast.literal_eval(data_dict[i]['standard_columns']) for i in datasets] for i in sublist])
        other_cols_counter = Counter([i for sublist in [ast.literal_eval(data_dict[i]['other_columns']) for i in datasets] for i in sublist])

    for i in INFORMATIONAL:
        if i in standard_cols_counter:
            standard_excluded.append((str(i),f'{i} ({standard_cols_counter[i]}/{ds_count}) (not a biological factor column)',standard_cols_counter[i]/ds_count))
            del standard_cols_counter[i]

    for i in RESPONSE_COLUMNS:
        if i in standard_cols_counter:
            standard_excluded.append((str(i),f'{i} ({standard_cols_counter[i]}/{ds_count}) (response column)',standard_cols_counter[i]/ds_count))
            del standard_cols_counter[i]

    if pretrained_val != None:

        # pipe in split to this, if specified in to be created and linked training parameters
        if mode == "Inference":
            pretrained_exclude = [i for i in standard_cols_counter if i not in pretrained_include]
            #import code
            #code.interact(local=dict(globals(), **locals()))
            if 'bio_columns' in model_dict[pretrained_val]:

                #exclude standard
                for i in pretrained_exclude:
                    if i not in standard_excluded:
                        standard_excluded.append((str(i),f'{i} ({standard_cols_counter[i]}/{ds_count}) (not used in pretrained model)',standard_cols_counter[i]/ds_count))
                    del standard_cols_counter[i]

                #exclude other cols
                pretrained_exclude = [i for i in other_cols_counter if i not in pretrained_include]
                for i in pretrained_exclude:
                    other_excluded.append((str(i),f'{i} ({other_cols_counter[i]}/{ds_count}) (not used in pretrained model)',other_cols_counter[i]/ds_count))
                    del other_cols_counter[i]

    else:
        pretrained_include = []

    #filter out id from standard columns display, where it never should be used in training.
    standard_cols_counts_display = [(str(x[0]),f"{x[0]} ({x[1]}/{ds_count}) {'(one hot encoded)' if STANDARD_COLUMN_NAMES[x[0]]['data_type'] == 'categorical' else ''} {' (included in pretrained model)' if x[0] in pretrained_include and mode == 'Fine-tuning' else ''}",x[1]/ds_count) for x in sorted(standard_cols_counter.items(), key = lambda x: x[1], reverse = True)]
    other_cols_counts_display = [(str(x[0]),f"{x[0]} ({x[1]}/{ds_count} {' (included in pretrained model)' if x[0] in pretrained_include and mode == 'Fine-tuning' else ''})",x[1]/ds_count) for x in sorted(other_cols_counter.items(), key = lambda x: x[1], reverse = True)]

    #wav_opts = [{"value":WN_STRING_NAME, "label":wav_str,"disabled":True if (valid_waves != ds_count or wavs_exclude) else False,
    #             'extra':(valid_waves/ds_count,ds_count-len(wave_counts)+1/ds_count)}]

    wav_opts = [{"value":WN_STRING_NAME, "label":wav_str,"disabled":True,
                 'extra':(valid_waves/ds_count,ds_count-len(wave_counts)+1/ds_count)}]

    #consolidate all values, and apply previous selections (as relevant)

    #disable, just mandate that it is true.
    wav_val = [] if (valid_waves != ds_count) else previous_selections['wav'] #previous_selections['wav']

    std_opts = [{"value":x[0],"label":x[1],"disabled":False,'extra':x[2]} for x in standard_cols_counts_display]+[{"value":x[0],"label":x[1],"disabled":True,'extra':x[2]} for x in standard_excluded]
    std_val = [m for m in [x[0] for x in standard_cols_counts_display] if m in previous_selections['std']]
    oc_opts = [{"value":x[0],"label":x[1],"disabled":False,'extra':x[2]} for x in other_cols_counts_display]+[{"value":x[0],"label":x[1],"disabled":True,'extra':x[2]} for x in other_excluded]
    oc_val = [m for m in [x[0] for x in other_cols_counts_display] if m in previous_selections['oc']]

    #import code
    #code.interact(local=dict(globals(), **locals()))


    children = [html.Div(children=[html.H4("Wave numbers:"),dcc.Checklist(id='data-pane-wav-numbers',
                  options=wav_opts,  # [9:]if f"datasets/" == i[:8]
                  value=wav_val,inputStyle={"margin-right":checklist_pixel_padding_between})] if ds_count != 0 else None), #if len(standard_cols_counts_display)>0 or len(other_cols_counts_display)>0 else None
            html.Div(id='wn_display_datasets',children=[html.Div(f"{m}: {round(float(ast.literal_eval(data_dict[m]['wave_number_min'])[0]),2)} - {round(float(ast.literal_eval(data_dict[m]['wave_number_max'])[0]),2)}; {round(float(ast.literal_eval(data_dict[m]['wave_number_step'])[0]),2)}") for m in datasets]),
            html.Div(children=[html.H4("Standard columns:"),dcc.Checklist(id='data-pane-columns-std',
                  options=std_opts,  # [9:]if f"datasets/" == i[:8]
                  value=std_val,inputStyle={"margin-right":checklist_pixel_padding_between})]) if (len(standard_cols_counts_display)+len(standard_excluded)) >0 else None,
            html.Div(children=[html.H4("Other columns:"),dcc.Checklist(id='data-pane-columns-oc',
                  options=oc_opts,  # [9:]if f"datasets/" == i[:8]
                  value=oc_val,inputStyle={"margin-right":checklist_pixel_padding_between})]) if (len(other_cols_counts_display)+len(other_excluded))>0 else None]

    return children

@callback(
    Output('pretrained_model_metadata_dict',"data", allow_duplicate=True),
    Input('pretrained-select','options'),
    Input('pretrained-select','value'),
    State('pretrained_model_metadata_dict',"data"),
    prevent_initial_call=True
)
def update_pretrained_metadata_dict(known_pretrained,selected_pretrained,pretrained_model_metadata_dict):

    if selected_pretrained == None:
        return pretrained_model_metadata_dict
    else:

        selected_pretrained = [selected_pretrained]

        km_set = set(known_pretrained)
        mmd_set = set(pretrained_model_metadata_dict)

        excess_metadata = mmd_set.difference(km_set)

        if len(excess_metadata) > 0:
            # remove out of date metadata
            for k in excess_metadata:
                print("deleting "+k)
                del pretrained_model_metadata_dict[k]

        uk_models = [k for k in selected_pretrained if k not in mmd_set]

        for k in uk_models:
            blob = STORAGE_CLIENT.bucket(DATA_BUCKET).get_blob(f'models/{k}')
            if blob == None:
                print("model does not exist in cloud storage - list not refreshed. bug?")
            elif blob.metadata != None:
                pretrained_model_metadata_dict.update({k: blob.metadata})
            else:
                data_bytes = blob.download_as_bytes()

                valid,_,data,metadata = data_check_models((k, "data:application/octet-stream;base64," + base64.b64encode(data_bytes).decode('utf-8')),load_model=True)

                if not valid:
                    metadata = {"no_metadata":1}
                    # flash a warning, but store in the metadata an indicator that the ds is not eligible

                # upload the flag to
                pretrained_model_metadata_dict.update({k:metadata})
                attach_metadata_to_blob(metadata, blob)

    return pretrained_model_metadata_dict

#refresh
@callback(
    Output('dataset-select','options', allow_duplicate=True),
    Output('dataset_select_last_clicked',"data", allow_duplicate=True),
    #Output('dataset-select',"value"),
    Input('refresh-ds', "n_clicks"),
    prevent_initial_call=True
)
def refresh_options(clicks):

    return get_datasets(),True

#change: input will be from dataset-select 'click trigger' to wipe behavior,
#and use the button as an input to dataset select
@callback(
    Output('data_metadata_dict',"data", allow_duplicate=True),
    Output('dataset_select_last_clicked', "data", allow_duplicate=True),
    Output("ds_ref_content","children", allow_duplicate=True),
    Input('dataset-select','options'),
    Input('dataset-select','value'),
    State('dataset_select_last_clicked', 'data'),
    #Input('refresh-ds', "n_clicks"),
    State('data_metadata_dict',"data"),
    #State('refresh-ds-count', "data"),
    prevent_initial_call=True
)
def update_data_metadata_dict(known_datasets,selected_datasets,click_trigger,data_metadata_dict):

    print(click_trigger)

    #if click_input == None:
    #    click_input = 0
    #print(click_input)
    #print(click_prev)

    #if refresh is hit, triggers special behavior where we flush known datasets
    if click_trigger: #click_input> click_prev:
        print('DS refresh button was clicked!')
        kd_set = set([i["value"] for i in get_datasets()])
        dmd_set = set()
        data_metadata_dict = {}
        increment = True
    else:
        increment = False
        kd_set = set([i["value"] for i in known_datasets])
        dmd_set = set(data_metadata_dict)

    excess_metadata = dmd_set.difference(kd_set)
    if len(excess_metadata) > 0:
        # remove out of date metadata
        for k in excess_metadata:
            del data_metadata_dict[k]

    uk_datasets = [k for k in selected_datasets if k not in dmd_set]
    for k in uk_datasets:
        blob = STORAGE_CLIENT.bucket(DATA_BUCKET).get_blob(f'datasets/{k}')
        if blob == None:
            print("dataset does not exist in cloud storage - list not refreshed. bug?")
        elif blob.metadata != None:
            data_metadata_dict.update({k:blob.metadata})
        else:
            #what this is supposed to do, is provide metadata if it wasn't uploaded through the app and autogenerated.
            data_bytes = blob.download_as_bytes()
            valid, message, _, metadata = data_check_datasets((k,"data:text/csv;base64,"+base64.b64encode(data_bytes).decode('utf-8')),load_data = False)
            if not valid:
                metadata = {k:{'ineligible':f"ineligible:{message}"}}
                #flash a warning, but store in the metadata an indicator that the ds is not eligible
                print('uh oh!') #output to dash warning that there are non-compatible datasets in the bucket not being
                #displayed. This will only flash the first time, after which the datasets will be viewable but will not
                #do anything when selected (should make a place in data pane to display this notice).

            #upload the flag to
            data_metadata_dict.update({k:metadata})
            attach_metadata_to_blob(metadata,blob)
    print("DMD:")
    print(data_metadata_dict)

    return data_metadata_dict,False,ref_content


@callback(
    Output('download-results', "data"),
    Input('btn-download-results', 'n_clicks'),
    State('run-name','value'),
    State('mode-select', 'value'),
    State('run_id', 'data'),
    State('dataset_titles', 'data'),
    State('params_dict', 'data')
)
def download_results(n_clicks,run_name,mode,run_id,dataset_titles,params_dict):

    if n_clicks is not None:

        #make a tarball
        #actually, I do want to use cloud storage instead of local for all files. Reason being is that the shared service
        #will write to shared files. I could use RUN_ID, but then I'd also need to figure out a way to schedule flushing
        #the folder versus just using the cloud storage lifecycle rule.

        tar_stream = io.BytesIO()

        with tarfile.open(fileobj=tar_stream, mode="w:gz") as tar:
            #stats
            blob = STORAGE_CLIENT.bucket(TMP_BUCKET).blob(f"stats_{run_id}.csv")

            obj = io.BytesIO(blob.download_as_bytes())
            obj.seek(0)
            info = tarfile.TarInfo(name = "stats.csv")
            info.size = len(obj.getvalue())
            tar.addfile(tarinfo = info,fileobj=obj)

            #config
            blob = STORAGE_CLIENT.bucket(TMP_BUCKET).blob(f"config_{run_id}.yml")

            obj = io.BytesIO(blob.download_as_bytes())
            obj.seek(0)
            info = tarfile.TarInfo(name="config.yml")
            info.size = len(obj.getvalue())
            tar.addfile(tarinfo=info, fileobj=obj)

            #add model object
            if params_dict["mode"] !="Inference":
                blob = STORAGE_CLIENT.bucket(TMP_BUCKET).blob(f"trained_model_{run_id}.hd5")

                obj = io.BytesIO(blob.download_as_bytes())
                obj.seek(0)
                info = tarfile.TarInfo(name=f"{[run_name if run_name is not None else run_id][0]}_trained_model.hd5")
                info.size = len(obj.getvalue())
                tar.addfile(tarinfo=info, fileobj=obj)

            #add data

            for i in dataset_titles:
                blob = STORAGE_CLIENT.bucket(TMP_BUCKET).blob(f"{i}_{run_id}.txt")

                obj = io.BytesIO(blob.download_as_bytes())
                obj.seek(0)
                info = tarfile.TarInfo(name=f"{i}.txt")
                info.size = len(obj.getvalue())
                tar.addfile(tarinfo=info, fileobj=obj)

        tar_stream.seek(0)

        return dcc.send_bytes(tar_stream.getvalue(),f"{run_id}_data.tar.gz")

#test: function to export values from data_columns after change



@callback(Output("columns_dict",'data', allow_duplicate=True),
          Input("data-pane-wav-numbers","value"),
          State("columns_dict", "data"),
        prevent_initial_call=True
          )
def update_wav_nums(wav,prev):

    prev['wav'] = wav
    #prev['wav']= set(wav)

    return prev

@callback(Output("columns_dict",'data', allow_duplicate=True),
          Input("data-pane-columns-std","value"),
          Input("data-pane-columns-std", "options"),
          State("columns_dict", "data"),
        prevent_initial_call=True
          )
def update_std_col(std,std_opts,prev):

    #remove if unselected
    for i in std_opts:
        print(i)
        if i['value'] in prev['std'] and i['value'] not in std and not i['disabled']:
            prev['std'].remove(i['value'])
    #add if selected
    for i in std:
        if i not in prev['std']:
            prev['std'].append(i)

    return prev

@callback(Output("columns_dict",'data', allow_duplicate=True),
          Input("data-pane-columns-oc","value"),
          Input("data-pane-columns-oc", "options"),
          State("columns_dict", "data"),
        prevent_initial_call=True
          )
def update_oc_col(oc,oc_opts,prev):

    #remove if unselected
    for i in oc_opts:
        if i['value'] in prev['oc'] and i['value'] not in oc:
            prev['oc'].remove(i['value'])
    #add if selected
    for i in oc:
        if i not in prev['oc']:
            prev['oc'].append(i)

    return prev

#function to recursively unpack the params_holder object to be able to get at nested parameters..
def unpack_children_for_values(out_dict,children):
    # tested for checklist and daq.toggleswitch- make sure as using new components this accounts for them correctly.

    if isinstance(children,dict):
        if 'value' in children:

            #if value is a list, possible to have multiple of them
            if 'options' in children:
                # if value is not a list, only one selection possible
                if not isinstance(children['value'], list):
                    out_dict[children['id']] = children['value']
                else:
                    for p in children['options']:
                        if isinstance(p,dict):
                            p = p['value']
                        if p in children['value']:
                            out_dict[(p,children['id'])] = True #maybe should store more data of this..
                        else:
                            out_dict[(p,children['id'])] = False #(children['id'],False)
            elif 'id' in children:
                #print(children)
                out_dict[children['id']] = children['value']

        for key in children:
            out_dict = unpack_children_for_values(out_dict,children[key])
    elif isinstance(children,list):
        for item in children:
            out_dict = unpack_children_for_values(out_dict, item)

    return out_dict

#version of this which assumes 'extra' metadata might be present
def unpack_data_pane_values_extra(out_dict,children):
    # tested for checklist and daq.toggleswitch- make sure as using new components this accounts for them correctly.

    if isinstance(children,dict):
        if 'value' in children:
            if 'options' in children:
                for p in children['options']:
                    if isinstance(p,dict):
                        out_dict[p['value']] = {}
                    if p['value'] in children['value']:
                        out_dict[p['value']]['selected'] = True
                    else:
                        out_dict[p['value']]['selected'] = False
                    out_dict[p['value']]['extra'] = p['extra']
        for key in children:
            out_dict = unpack_data_pane_values_extra(out_dict,children[key])
    elif isinstance(children,list):
        for item in children:
            out_dict = unpack_data_pane_values_extra(out_dict, item)

    return out_dict


#considering: make instead of reading from params dict, have it loop through different blocks
#and populate the params dict here. f
@callback(
    Output('alert-model-run-processing-fail','is_open'),
     Output('alert-model-run-ds-fail','is_open'),
     Output('alert-model-run-models-fail','is_open'),
     Output('run-message', 'children'),
     Output('config-report', 'children'),
     Output('stats-out', 'children'),
     Output('artifacts-out', 'children'),
     Output('download-out', 'children'),
     Output('upload-out', 'children', allow_duplicate=True),
     Output("params_dict","data"),
     Output("dataset_titles", "data"),
     Output("run_id", "data"),
     Input('run-button', 'n_clicks'),
     State('mode-select', 'value'),
     State('pretrained_value_dict', 'data'),
     State('pretrained_model_metadata_dict', 'data'),
     State('approaches_value_dict', 'data'),
     State('data-pane', 'children'),
     State('dataset-select', 'value'),
     State('params-holder', 'children'),
     State("dataset_titles", "data"),
     State("session-id", "data"),
     State('data_metadata_dict',"data"),
     #running=[
     #   (Output("run-button", "disabled"), True, False),
     # ],
    prevent_initial_call=True,
    #background = True #todo this is returning pickle/diskcache error trying to run as background task. Confirmed that nothing in the callback python function itself is causing the error
    #todo Next, try to run a simpler background task to see if the libraries and background are at least correctly configured.
 )
def model_run_event(n_clicks,mode,pretrained_model,pretrained_model_metadata,approach,columns,datasets,params_holder,dataset_titles,session_id,data_metadata):

    pretrained_model = pretrained_model['value']
    approach = approach['value']

    #loop through and look for "values". if exists options, use values to determine true or false.
    #Use id as the parameter name, make sure that convention is kept in get_params

    #this object enables us to use the 'selected' and 'extra' (for wavs, (%valid,%equivalent) and for others % present in total ds) and assess whether it is fine, warning, or error
    data_pane_vals_dict = unpack_data_pane_values_extra({}, columns)

    if n_clicks is not None:

        message = "Run Failed: "

        processing_fail = False
        ds_fail = False
        model_fail = False

        data, artifacts, stats = [], [], []

        any_error = False

        if len(datasets)==0:

            message = message + "no datasets specified"
            ds_fail = True
            config_out_payload = []

            any_error = True

            #payload = False,True,False,"Run Failed: ",[]

        #make this specific depending on mode.
        if (pretrained_model is None and mode !="Training") or (approach is None and mode!="Inference"):
            if any_error:
                message = message + " & no training approach or pretrained model specified"
            else:
                message = message + "no training approach or pretrained model specified"
            model_fail = True

            any_error = True
            config_out_payload = []
            #payload = False,False,False,"Run Failed: no model specified",[]

        if not any([data_pane_vals_dict[n]['selected'] for n in data_pane_vals_dict]):
            if any_error:
                message = message + " & data error: no valid columns for operation"
            else:
                message = message + "data error: no valid columns for operation"

            ds_fail = True

            any_error = True
            config_out_payload = []

        if not data_pane_vals_dict[WN_STRING_NAME]['selected']:

            if any_error:
                message = message + f" & data error: no valid {WN_STRING_NAME} for operation"
            else:
                message = message + f"data error: no valid {WN_STRING_NAME} for operation"

            ds_fail = True

            any_error = True
            config_out_payload = []

        #print(params_dict)

        run_id = ""  # set default so if it errors out, still can return outputs
        if not any_error:
            try:

                params_dict = unpack_children_for_values({},params_holder)

                params_dict['mode'] = mode

                # time.time() makes it unique even when parameters are fixed, may want to change behavior later
                run_id = str(abs(hash("".join(["".join(params_dict), str(mode), str(approach if approach!=None else ""), str(pretrained_model if pretrained_model!=None else ""), "".join(datasets)])+str(time.time()))))

                config_dict = {'params_dict':params_dict}

                config_dict.update({"Datasets":",".join(datasets)})
                config_dict.update({"Pretrained model": pretrained_model})
                config_dict.update({"Training approach": approach})
                config_dict.update({"Mode":mode})

                config_table = pd.DataFrame(config_dict,index=[0])

                blob = STORAGE_CLIENT.bucket(TMP_BUCKET).blob(f'config_{run_id}.yml')
                blob.upload_from_string(config_table.to_csv(), 'text/csv')

                config_out_children = [html.Div(id='config-body',children=[html.Div(id='run-name-block',children =[html.Div(id='run-name-prompt',children = "Run name:"),
                                        dcc.Input(id='run-name', type="text", placeholder="my_unique_pretrained_model_name",style={'textAlign': 'left', 'vertical-align': 'top','width':"150%"}),
                                                                               html.Div(id='description-prompt', children="Description:"),
                                        dcc.Textarea(id='description',style={'textAlign': 'left','vertical-align': 'top','height':100,'width':"150%"})
                                                                                ],style={"display": "inline-block"}) if mode!="Inference" else html.Div(id='run-name-block',children=[html.Div(id='run-name'),html.Div(id='description')])] +\
                                       [html.Div(id='config-report-rc',children = "Run Configuration:"),
                                       html.Div(id='config-report-mode', children="Mode: "+ mode),
                                       html.Div(id='config-report-model',children="Pretrained model: " + str(pretrained_model)) if mode != "Training" else None,
                                       html.Div(id='config-report-model',children="Training approach: " + str(approach)) if mode != "Inference" else None,
                                       html.Div(id='config-report-datasets',children ='Datasets: ')] +\
                                       [html.Div(id='config-report-datasets-' + i,children="- "+str(i),style={'marginLeft': 15}) for i in datasets] +\
                                       [html.Div(id='config-report-parameters',children ='Parameters: ')] + \
                                       [html.Div(id='config-report-parameters-' + a,children="- "+a+": "+str(b),style={'marginLeft': 15}) for (a,b) in config_dict["params_dict"].items()],style={'width': left_body_width})]


                #this is where model will actually run
                #if mode == "Training":

                #CONSIDER CACHING DATASETS AND MODELS ON DISK FOR FASTER RETRIEVAL!

                #just write it here to take advantage of variables, turn into fxns later as sensible.

                #at some point, have this read from a cache on disk to speed up.

                print("GOT TO HERE")

                data = []
                for m in datasets:
                    cloud_download = False
                    #use the metadata dict to get hash
                    if 'data_hash' in data_metadata[m]:
                        ds_hash = data_metadata[m]['data_hash']
                        if ds_hash in CACHE:
                            LOGGER_MANUAL.info(f"{session_id} Reading data {m} from local cache (rid: {run_id[:6]}...)")

                            data_b64 = CACHE.get(data_metadata[m]['data_hash'])
                        else:
                            cloud_download = True
                    else:
                        cloud_download = True

                    #if cloud_download == True:

                    if cloud_download:
                        LOGGER_MANUAL.info(f"{session_id} Reading data {m} from cloud (rid: {run_id[:6]}...)")
                        data_b64 = STORAGE_CLIENT.bucket(DATA_BUCKET).blob(f"datasets/{m}").download_as_bytes()
                        blob = STORAGE_CLIENT.bucket(DATA_BUCKET).get_blob(f'datasets/{m}')
                        blob.metadata['data_hash']
                        CACHE.set(blob.metadata['data_hash'], data_b64, expire= CACHE_EXPIRY * 24 * 60 * 60)

                    data.append(pd.read_csv(io.BytesIO(data_b64)))

                #data = [pd.read_csv(io.BytesIO(STORAGE_CLIENT.bucket(DATA_BUCKET).blob(f"datasets/{i}").download_as_bytes())) for i in datasets]

                if len(data)==1:
                    data = data[0]

                if mode != "Inference":
                    #use params to populate splitvec is specified.
                    if config_dict["params_dict"]['define-splits']==False:
                        splitvec = None
                    else:
                        splitvec = config_dict["params_dict"]['splits-slider']
                else:
                    splitvec = [0,0]

                total_bio_columns = 100 #expose later as a parameter, would need to only be available in 'training' mode, as fine tuning will have fixed architecture.

                if mode != 'Inference':

                    callbacks = [EpochCounterCallback(session_id, LOGGER_MANUAL, run_id)]

                    if config_dict["params_dict"].get("define-early-stopping"):
                        patience = config_dict["params_dict"].get("early-stopping-patience",4)
                        metric = config_dict["params_dict"].get("early-stopping-metric","val_loss")
                        callbacks.append(EarlyStopping(monitor=metric, patience=patience, verbose=1, restore_best_weights=True))
                else:
                    callbacks = []

                # if there is custom interpolation needed, extract it.
                if config_dict["params_dict"]['define-interp']:
                    interp = [float(config_dict["params_dict"]["interp-min"]), float(config_dict["params_dict"]["interp-max"]),
                              float(config_dict["params_dict"]["interp-step"])]
                else:
                    interp = None

                drop_cols = [i for i in data_pane_vals_dict if not data_pane_vals_dict[i]['selected'] and (
                        i not in INFORMATIONAL and i not in RESPONSE_COLUMNS and i not in WN_STRING_NAME)]

                if isinstance(data, list):
                    data = [i.drop(columns=drop_cols, errors="ignore") for i in data]
                else:
                    data = data.drop(columns=drop_cols, errors="ignore") #need to ignore errors as sometimes data_pane_vals_dict provides columns only present in pretrained model, not dataset.



                if mode == 'Training':

                    print("GOT TO HERE2")

                    if 'parameters' in TRAINING_APPROACHES[approach]:
                        supplied_params = {a:config_dict["params_dict"][a] for a in TRAINING_APPROACHES[approach]['parameters'] if a in config_dict["params_dict"]}

                    #import code
                    #code.interact(local=dict(globals(), **locals()))

                    #LOGGER_MANUAL.info(f"{session_id} Removing dropped columns (rid: {run_id[:6]}...)")
                    #drop before formatting so that metadata lines up.

                    LOGGER_MANUAL.info(f"{session_id} Preprocessing and formatting data (rid: {run_id[:6]}...)")

                    formatted_data, format_metadata, _ = format_data(data,filter_CHOICE=config_dict["params_dict"]["wn-filter"],scaler=config_dict["params_dict"]["bio-scaling"],\
                                                                     wn_scaler=config_dict["params_dict"].get("wn-scaling"),\
                                                                     response_scaler=config_dict["params_dict"].get("response-scaling"),\
                                                                     splitvec=splitvec, interp_minmaxstep=interp)

                    #remove any columns that were unselected in data pane.

                    LOGGER_MANUAL.info(f"{session_id} Starting model training (rid: {run_id[:6]}...)")

                    if approach == 'Basic model':
                        #supplying the parameters, using default values
                        model, training_outputs, additional_outputs = TrainingModeWithoutHyperband(
                            data=formatted_data,
                            bio_idx=format_metadata["datatype_indices"]["bio_indices"],
                            wn_idx=format_metadata["datatype_indices"]["wn_indices"],
                            total_bio_columns=total_bio_columns,
                            epochs = config_dict["params_dict"].get('epoch',30),
                            **supplied_params,
                            callbacks = callbacks,
                            seed_value = config_dict["params_dict"].get('seed')
                        )

                    elif approach == 'hyperband tuning model':
                        model, training_outputs, additional_outputs = TrainingModeWithHyperband(
                            data=formatted_data,
                            bio_idx=format_metadata["datatype_indices"]["bio_indices"],
                            wn_idx=format_metadata["datatype_indices"]["wn_indices"],
                            total_bio_columns=total_bio_columns,
                            epochs=config_dict["params_dict"].get('epoch', 30),
                            max_epochs=config_dict["params_dict"].get('hyperband_max_epoch', TRAINING_APPROACHES[approach]["parameters"]["hyperband_max_epoch"]["default_value"]), #, #make this into general training parameter.
                            callbacks = callbacks,
                            seed_value = config_dict["params_dict"].get('seed')
                        )

                    LOGGER_MANUAL.info(f"{session_id} Finished model training (rid: {run_id[:6]}...)")

                else:
                    #for now, decided it's quick + easy to just download object from cloud each  time, can build in caching if that changes.
                    LOGGER_MANUAL.info(f"{session_id} Reading model from cloud (rid: {run_id[:6]}...)")
                    model, metadata, _ = extract_model(STORAGE_CLIENT.bucket(DATA_BUCKET).blob(f"models/{pretrained_model}").download_as_bytes(),pretrained_model, upload_to_gcp=False)
                    blob = STORAGE_CLIENT.bucket(DATA_BUCKET).get_blob(f'models/{pretrained_model}')
                    obj_metadata = blob.metadata

                    #make metadata a list if it's not already for easier assumptions.

                    if not isinstance(metadata,list):

                        metadata = [metadata]

                    LOGGER_MANUAL.info(f"{session_id} Preprocessing and formatting data (rid: {run_id[:6]}...)")

                    formatted_data, fd_outputs, _ = format_data(data, filter_CHOICE=metadata[-1]['filter'],
                                                           scaler=metadata[-1]['scaler'], splitvec=[0, 0] if mode == "Inference" else splitvec,
                                                           interp_minmaxstep=interp,add_scale=True if mode == "Fine-tuning" else False)

                if mode == "Inference":

                    LOGGER_MANUAL.info(f"{session_id} Generating predictions (rid: {run_id[:6]}...)")
                    predictions = InferenceMode(model, formatted_data, metadata[-1]['scaler'],metadata[-1]['model_col_names'])
                    LOGGER_MANUAL.info(f"{session_id} Finished generating predictions (rid: {run_id[:6]}...)")

                elif mode == "Fine-tuning":

                    #import code
                    #code.interact(local=dict(globals(), **locals()))

                    LOGGER_MANUAL.info(f"{session_id} Fine-tuning model (rid: {run_id[:6]}...)")

                    model2, training_outputs_finetuning, additional_outputs_finetuning = TrainingModeFinetuning(
                        model=model, data=formatted_data,
                        bio_idx=fd_outputs["datatype_indices"]["bio_indices"],
                        names_ordered=metadata[-1]['model_col_names'],
                        seed_value=config_dict["params_dict"].get('seed'),
                        epochs=config_dict["params_dict"].get('epoch', 30),
                        callbacks = callbacks,)

                    LOGGER_MANUAL.info(f"{session_id} Finished fine-tuning model (rid: {run_id[:6]}...)")

                elif mode == "Evaluation":
                    pass

                #data,artifacts,stats,dataset_titles = run_model(

                message = "Run Succeeded!"

                config_out_payload = config_out_children

            except Exception as e:
                message = "Run Failed: error while processing algorithm"
                config_out_payload = [html.Div(id='error-title',children="ERROR:"),html.Div(id='error message',children =[str(e)])]
                processing_fail = True


        download_out = [html.Div([html.Button("Download Results", id="btn-download-results"),
                    dcc.Download(id="download-results")]) if not (processing_fail or any_error) else ""]

        artifacts_out = html.Div(artifacts,style={'textAlign': 'left', 'vertical-align': 'top'})

        stats_out = html.Div([
                        html.Div(
                        [
                            html.Div(id='stats-title',children="Run stats:"),
                            dash_table.DataTable(stats.to_dict('records')),
                        ],style={'textAlign': 'left', 'vertical-align': 'top'}) if not (processing_fail or any_error) else ""])

        return [processing_fail,ds_fail,model_fail,message,config_out_payload,stats_out,artifacts_out,download_out,html.Button("Upload Trained model", id="btn-upload-pretrained") if mode != "Inference" and not (processing_fail or any_error) else "",params_dict if not (processing_fail or any_error) else "",dataset_titles,run_id]


@callback(Output('train_val',"children"),
                Output('val_val',"children"),
                Output('test_val', "children"),
                Input('splits-slider', 'value')
          )
def populate_splits_fields(slider_val):

   return f"Train (%): {slider_val[0]}",f"Validation (%): {slider_val[1]-slider_val[0]}",f"Test (%): {100-slider_val[1]}"
@callback(Output('splits_status',"children"),
                Output('splits_choice',"children"),
                Input('define-splits', 'value')
          )
def populate_split_selection(splits_val):

    return f"Define splits: {splits_val}", [html.Div(id='train_val'),html.Div(id='val_val'),html.Div(id='test_val'),
                                            dcc.RangeSlider(0, 100, 1, marks = {x*5:x*5 for x in range(20)}, value=[60, 80], id='splits-slider',
                                                            allowCross=False)]  if splits_val else None


@callback(Output('interp_status',"children"),
                Output('interp-choice',"children"),
                Input('define-interp', 'value')
          )
def populate_interp_selection(interp_val):

    return f"Custom interpolation: {interp_val}", [html.Div("Min:"),dcc.Input(id="interp-min", type="number", placeholder="float (ex. 3999.0)"),
                                            html.Div("Max:"),dcc.Input(id="interp-max", type="number", placeholder="float (ex. 11476.0)"),
                                            html.Div("Step:"),dcc.Input(id="interp-step", type="number", placeholder="float (ex. 8.0)")] if interp_val else None

@callback(Output('early_stopping_status',"children"),
                Output('early_stopping_choice',"children"),
                Input('define-early-stopping', 'value')
          )
def populate_early_stopping_selection(early_stopping_val):

    return f"Enable Early Stopping: {early_stopping_val}", [html.Div("Metric:"),dcc.Input(id="early-stopping-metric", type="text", placeholder="val_loss"),
                                            html.Div("Max:"),dcc.Input(id="early-stopping-patience", type="number", placeholder="int (ex 4)")] if early_stopping_val else None


#@callback(
#                    Output('preprocessing_params', 'children',allow_duplicate=True),
#                    Input('mode-select', 'value'),
#                    Input('approaches_value_dict', 'data'),
#                    State('preprocessing_params', 'children'),
#    prevent_initial_call=True
#)
#def get_preprocess_parameters(mode,approach,previous):
#
#    approach = approach['value']
#
#    if mode == 'Training' and approach is not None:
#        return [html.H4("Preprocessing"), preprocessing_params]
#    elif mode == 'Inference':
#        return None
#    else:
#        return previous

@callback(
                    Output('mode_params', 'children'),
                    Input('mode-select', 'value'),
                    State('mode_params', 'children'))
def get_mode_parameters(mode,previous):

    #if mode != "Inference" and previous != "Inference" and previous !=[]:
    #    return previous
    if mode != "Inference":
        return [
            html.H4("Training"),
            html.Div("Seed:"),
            dcc.Input(id="seed", type="number", placeholder="int"),
            html.Div(id='splits_status', children='Define splits: False'),
            daq.ToggleSwitch(id='define-splits', value=False, style={"width": "50%"}),
            html.Div(id='splits_choice'),
            html.Div('Number of epochs'),
            dcc.Input(id="epoch", type="number", placeholder="int (default: 30)"),
            html.Div(id='early_stopping_status', children='Enable Early Stopping: False'),
            daq.ToggleSwitch(id='define-early-stopping', value=False, style={"width": "50%"}),
            html.Div(id='early_stopping_choice')
        ]
    else:
        return [html.H4("Inference")]

@callback(
                    Output('approach_params', 'children'),
                    Input('mode-select', 'value'),
                    Input('approaches_value_dict', 'data'),
                    #State("approach_params",'children'),
    prevent_initial_call=True
)
def get_approach_parameters(mode,approach):

    approach = approach['value']

    if mode == 'Training' and approach is not None:
        print("in here")
        return  [html.H4(approach),TRAINING_APPROACHES[approach]["dash_params"] if "dash_params" in TRAINING_APPROACHES[approach] else None]

@callback(
    Output('mode-select-output', 'children'),
    Input('mode-select', 'value')
)
def update_output(mode):

    if mode == "Training":
        definition = "Create a model from scratch using a training approach"
    if mode == "Inference":
        definition = "Predict ages with a pretrained model without needing age information"
    if mode == "Fine-tuning":
        definition = "Fine-tune a pretrained model for better performance on smaller datasets"
    return definition

def extract_metadata(file_object):

    if 'metadata' not in file_object:
        return None
    else:
        metadata = file_object['metadata']

        # convert to pyton dict:

        # common formatting may be desirable over below:
        metadata = {key: (metadata[key][()].decode('utf-8') if metadata[key].shape == () else (', ').join(
            [x.decode('utf-8') for x in list(metadata[key])])) for key in metadata}

        return metadata
def attach_metadata_to_blob(metadata,blob):

    # declare the metdata on the object in cloud storage.
    blob.metadata = metadata

    blob.patch()

def attach_null_metadata(blob):
    # declare the metdata on the object in cloud storage.
    blob.metadata = {"no_metadata":1}

    blob.patch()

@callback(
    Output("data_metadata_dict", "data", allow_duplicate=True),
    Output("alert-dataset-success", "is_open"),
    Output("alert-dataset-fail", "is_open"),
    Output("alert-dataset-fail", "children"),
    Output("upload-ds", "contents"), #this output is workaround for known dcc.Upload bug where duplicate uploads don't trigger change.
    Output("dataset-select","options"),
    Output("ds_ref_content", "children", allow_duplicate=True),
    Output("dataset_select_last_clicked", "data"),
    Input('upload-ds', 'filename'),Input('upload-ds', 'contents'),
    State('data_metadata_dict','data'),
    prevent_initial_call=True
)
def datasets_to_gcp(filename,contents,data_metadata_dict):

    #this gets called by alert on startup as this is an output, make sure to not trigger alert in this case
    if not filename == None and not contents == None:
        for i in zip(filename,contents):
            valid, message,data,metadata = data_check_datasets(i,load_data=True)

            if valid:
                print('uploading')

                #todo: loop through metadata for existing datasets to make sure a dataset of the same hash doesn't already exist
                #todo: or, display hash to user along with other ds metadata

                blob = STORAGE_CLIENT.bucket(DATA_BUCKET).blob(f'datasets/{i[0]}')

                blob.upload_from_string(data, 'text/csv')

                #cache the dataset

                CACHE.set(metadata['data_hash'],data,expire= CACHE_EXPIRY * 24 * 60 * 60) #two week expiry

                attach_metadata_to_blob(metadata,blob)

                #if filename is already in data_metadata_dict, clear it so that the contents will be properly
                #pulled

                if i[0] in data_metadata_dict:
                    print('clearing data_metadata_dict on reupload')
                    del data_metadata_dict[i[0]]

                message = ""


            else:
                valid = False
    else:
        return data_metadata_dict,None,None,None,None,get_datasets(),False,ref_content

    return data_metadata_dict,valid,not valid,message,None,get_datasets(),False,ref_content

#similar to the one below,

@callback(
    Output("model-upload-from-session-success", "is_open"),
    Output("model-upload-from-session-failure", "is_open"),
    Input('btn-upload-pretrained', 'n_clicks'),
    State('run-name',"value"),
    State('description', "value"),
    State("run_id","data")
)
def trained_model_publish(n_clicks,run_name,description,run_id):

    #later may want to define try / or failure condition

    if n_clicks is not None:
        if run_name is None:
            return False,True
        # TODO check for unique
        #TODO check for disallowed characters
        elif False:
            pass

        blob = STORAGE_CLIENT.bucket(TMP_BUCKET).blob(f"trained_model_{run_id}.hd5")

        STORAGE_CLIENT.bucket(TMP_BUCKET).copy_blob(blob,STORAGE_CLIENT.bucket(DATA_BUCKET) , f'models/{run_name}.hd5')

        #TODO: attach description and other metadata to published model object.

        return True,False

@callback(
    Output("alert-model-success", "is_open"),
    Output("alert-model-fail", "is_open"),
    Output("alert-model-fail", "children"),
    Input('upload-pretrained', 'filename'),Input('upload-pretrained', 'contents')
)
def models_to_gcp(filenames, contents):

    success = True
    # this gets called by alert on startup as this is an output, make sure to not trigger alert in this case
    if not filenames == None and not contents == None:
        for i in zip(filenames, contents):

            if i[0].endswith('.zip'):
                valid = True
                message = None
            else:
                valid = False
                message = "not named like a .zip file"

            if valid:

                content_type, content_string = i[1].split(',')

                decoded = base64.b64decode(content_string)

                model, metadata,blob = extract_model(decoded,i[0],upload_to_gcp=True)

                #attach the full metadata file as a .pickle. Also, attach particular accessory metadata that will
                #be needed later: all columns used in current model, wn attributes, description... parameter values?
                if metadata is not None:

                    if isinstance(metadata,list):
                        num_trainings = len(metadata)

                        metadata = metadata[-1]
                    else:
                        num_trainings = 1

                    _, _, wave_number_min, wave_number_step, wave_number_max, _ = wnExtract(metadata['model_col_names']["wn_columns_names_ordered"])

                    #for scaler, iterate through column transformer until we hit wn**. prior to it, we will determine if there is a single scaler, or else will report 'mixed' for biological.
                    #for wn

                    wn_scaler = str(metadata['scaler'][0].named_transformers_[WN_STRING_NAME])[:-2]
                    response_scaler = str(metadata['scaler'][0].named_transformers_[RESPONSE_COLUMNS[0]])[:-2]
                    bio_scalers= [str(metadata['scaler'][0].named_transformers_[i])[:-2] for i in metadata['scaler'][0].named_transformers_ if i != WN_STRING_NAME or i != response_scaler]
                    bio_scaler = bio_scalers[0] if all([i==bio_scalers[0] for i in bio_scalers]) else 'mixed'

                    wn_filter = metadata['filter']

                    print('attaching metadata')
                    custom_metadata = {'num_trainings':num_trainings,'description':metadata['description'],"max_bio_columns":len(metadata['model_col_names']["bio_column_names_ordered_padded"]),\
                    'wave_number_min':wave_number_min,'wave_number_max':wave_number_max,"wave_number_step":wave_number_step,'bio_columns':metadata['model_col_names']["bio_column_names_ordered"],\
                    'wn_scaler':wn_scaler,'response_scaler':response_scaler,'bio_scaler':bio_scaler,'wn_filter':wn_filter}

                    print(custom_metadata)

                    attach_metadata_to_blob(custom_metadata, blob)

                else:
                    attach_null_metadata(blob)

            else:
                success = False

        return success, not success, message
    else:
        return False, False, None

def extract_model(model_zip_bytes,model_zip_name,upload_to_gcp=False):

    blob = STORAGE_CLIENT.bucket(DATA_BUCKET).blob(f'models/{model_zip_name}')

    with tempfile.NamedTemporaryFile(delete=True, suffix=".zip") as temp_zip:
        temp_zip.write(model_zip_bytes)
        model, metadata = loadModelWithMetadata(temp_zip, model_zip_name[:-4])
        if upload_to_gcp:
            blob.upload_from_string(model_zip_bytes)
            #blob.upload_from_filename(temp_zip.name)

    return model, metadata, blob


def data_check_datasets(file,load_data = True):
    message = "Dataset upload unsuccessful: did not pass standards checking"

    valid = True

    #to check - presence of duplicated rows (reject).

    if not ".csv" == file[0][-4:]:
        return False,message + "; - file not named as a csv",None,None
    else:
        content_type, content_string = file[1].split(',')
        decoded = base64.b64decode(content_string)

        # assemble dataset metadata object.
        # extract column names- interpret wav #s from other columns based on known rules.

        # get the dataset hash, which will be used to validate it as the same ds object and help improve performance with caching.
        # I could do this within data check datasets, but that will mean that each dataset will be redowloaded
        #pd.read_csv(io.BytesIO(STORAGE_CLIENT.bucket(DATA_BUCKET).blob(f"datasets/{i}").download_as_bytes()))
        #for i in datasets]
        #ds_hash = hash_dataset

        #print("loaded")



        data = io.StringIO(decoded.decode('utf-8'))

        # check that the dataset does not contain duplicated columns. pd automatically renames duplicate columns so cannot do this
        print('checking duplicate column names')
        reader = csv.reader(data)
        columns = next(reader)

        # check that column names are unique
        if len(columns) != len(set(columns)):
            valid = False
            message = message + "; - " + f"columns must be uniquely named"
        dataset  =pd.read_csv(data)
        data_hash = hash_dataset(dataset)

        # avoid behavior of pandas to rename duplicate names


        # along with parsing columns, rename the column metadata to 'standard' names. For example,
        # the identifer field, no matter what it is called, will be replaced by id. Have an visual indicator
        # for columns that cannot be matched to global naming.

        # for now, let's just hardcode a dict. have it as a seperate file, 'standard_variables.py'

        if 'metadata' in columns:
            pass
            # in this case, use provided info to extract the wave numbers by position. assume the wav numbers
            # to be the rightmost columns in consecutive order according to the metadata information.
        elif any('wn' in c for c in columns):
            # in this case, use string matching to determine the wave numbers
            # use the values from the matched strings to arrive at the relevant variable values.
            matches = ['wn' in c for c in columns]

            wave_number_start_index = matches.index(True)
            wave_number_end_index = len(matches) - 1 - matches[::-1].index(True)

            a, b = columns[wave_number_start_index][2:], columns[wave_number_end_index][2:]
            # make the metadata terms for 'max' and 'min' come from # values instead of assuming low/high high/low ordering
            if float(a) > float(b):
                wave_number_max = a
                wave_number_min = b
            else:
                wave_number_max = b
                wave_number_min = a

            wave_number_step = (float(wave_number_max) - float(wave_number_min)) / (
                        wave_number_end_index - wave_number_start_index)

            #not robust to wn being at start or end of df
            #other_cols_start = 0  # assume this for now, but see if there are exeptions
            #other_cols_end = wave_number_start_index - 1
            #non_wave_columns = columns[other_cols_start:other_cols_end]

            non_wave_columns = [i for i in columns if 'wn' not in i]
        else:
            non_wave_columns = columns
            wave_number_start_index = "-1"
            wave_number_end_index = "-1"
            wave_number_min = "-1"
            wave_number_max = "-1"
            wave_number_step = "-1"

        # classify provided columns into standard and other columns:

        standard_cols = []
        other_cols = []



        for f in non_wave_columns:
            if f in STANDARD_COLUMN_NAMES:
                standard_cols.append(f)
            else:
                other_cols.append(f)

        # If wav numbers are absent in dataset, supply relevant metadata fields with -1
        metadata = {"standard_columns": standard_cols,
                    "other_columns": other_cols, "wave_number_start_index": [wave_number_start_index],
                    "wave_number_end_index": [wave_number_end_index], "wave_number_min": [wave_number_min],
                    "wave_number_max": [wave_number_max], "wave_number_step": [wave_number_step],
                    "data_hash":data_hash,'data_rows':len(dataset)}
        if valid:
            return True,"",decoded if load_data else None,metadata #
        else:
            return False,message,None,None

def data_check_models(file,load_model):

    #unpack the zip into tempdir.

    valid = True
    message = None

    #check that it's named correctly (very casually)
    if not ".hd5" == file[0][-4:] and not ".h5" == file[0][-3:] and not ".hdf5" == file[0][-5:]:
        valid = False
        message = "not a correctly named file"
        decoded = None
        metadata = None
    else:
        content_type, content_string = file[1].split(',')
        decoded = base64.b64decode(content_string)
        # check its the correct type of file

        try:
            with h5py.File(io.BytesIO(decoded), 'r') as f:
                metadata = extract_metadata(f)
        except Exception as e:
            valid = False
            message = f"could not read file; {e}"
            decoded = None
            metadata = None

    return valid,message,decoded if load_model else None,metadata

    #check that it has the essential metadata fields

def run_model(mode,model,datasets,run_id):

    start = time.time()

    if mode == 'Inference':

        data = [[rand.randint(1,15) for i in range(100)]]
        titles = ['Data: Ages'] #go.Scatter(x=list(range(len(data))), y=data)
        graphs = [html.Div(id='figure-title', style={'textAlign': 'left'}, children=titles[0]),dcc.Graph(id='figure',figure = go.Figure(data=[go.Histogram(x=data[0])]).update_layout(margin=dict(l=20, r=20, t=20, b=20)))]

        #question- show dynamic graphs or pngs? probably depends on real artifacts

        stats = {'max_cpu_used':[rand.randint(0,100)],'max_memory_used':[rand.randint(0,256)],'time_elapsed':["{:.2f}".format(time.time() - start)]}

    else:
        data = [[[1,2,3],[1,0.97,0.2]]]
        titles = ['Data: Performance curve']
        graphs = [html.Div(id='figure-title', style={'textAlign': 'left'}, children=titles[0]),dcc.Graph(id='figure',figure = go.Figure(data=[go.Scatter(x=data[0][0], y=data[0][1])]).update_layout(margin=dict(l=20, r=20, t=20, b=20)))]
        stats = {'Accuracy':["{:.2f}".format(rand.randint(0,100)/100)],'Recall':["{:.2f}".format(rand.randint(0,100)/100)],'AUC':["{:.2f}".format(rand.randint(0,100)/100)],'time_elapsed':["{:.2f}".format(time.time() - start)]}

        #write out model object
        #observation- be careful with lifecycle. Object is written to temp data, but for longer running jobs need to make sure the object
        #will not be cycled from temp data prior to publication.
        blob = STORAGE_CLIENT.bucket(TMP_BUCKET).blob(f"trained_model_{run_id}.hd5")
        blob.upload_from_string(pd.DataFrame([1,2,3]).to_csv(), 'text/csv')

    stats_table = pd.DataFrame.from_dict(stats)

    # write out data, artifacts, stats_table to tmp files
    blob = STORAGE_CLIENT.bucket(TMP_BUCKET).blob(f'stats_{run_id}.csv')
    blob.upload_from_string(stats_table.to_csv(), 'text/csv')

    artifacts = [html.Div(id='figure-area', children=graphs)]

    #save plot data to tmp files
    for i in range(len(titles)):
        blob = STORAGE_CLIENT.bucket(TMP_BUCKET).blob(f"{titles[i]}_{run_id}.txt")
        blob.upload_from_string(pd.DataFrame(data[i]).to_csv(), 'text/csv')

    return data, artifacts, stats_table, titles

