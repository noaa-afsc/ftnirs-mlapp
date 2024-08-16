import csv
from collections import Counter
import json
import os
import tarfile
import io
import ast
import dash
import dash_daq as daq
import plotly.graph_objects as go
#from dash.dependencies import Input, Output
from dash import dcc,html, dash_table,callback_context, callback,  Output, Input, State, DiskcacheManager #,State
#from dash_extensions.enrich import Output, Input, State, DiskCacheManager #not sure if I need this yet. Maybe, these (diskcache) got rolled into vanilla diskcache?
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import pandas as pd
import random as rand
from google.cloud import storage
import plotly.graph_objects as go
import time
import base64
import h5py
import app_data
import diskcache
#safer and less bespoke than what I previously implemented.
import uuid
from dotenv import load_dotenv

load_dotenv('./tmp/.env')

dash.register_page(__name__, path='/',name = os.getenv("APPNAME")) #

#declare diskcache, to manage global variables/long running jobs in a way that
#is multiple user and thread safe and fault tolerant.

cache = diskcache.Cache("./cache")

#this should be an .env variable
GCP_PROJ="ggn-nmfs-afscftnirs-dev-97fc"

#STORAGE_CLIENT = storage.Client(project=os.getenv("GCP_PROJ"))
STORAGE_CLIENT = storage.Client(project=GCP_PROJ)
DATA_BUCKET = os.getenv("DATA_BUCKET")
TMP_BUCKET = os.getenv("TMP_BUCKET")

def get_objs():
    objs = list(STORAGE_CLIENT.list_blobs(DATA_BUCKET))
    return [i._properties["name"] for i in objs]

#should have this return elegible datasets as 1st tuple inelibile as 2nd tuple. (requires metadata check every call... )?
def get_datasets():
    return [i[9:] for i in get_objs() if "datasets/" == i[:9]]

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
BUTTON_DEFAULT_STYLE = {"font-weight": 900,"width":100,"height":100}

layout = html.Div(id='parent', children=[

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
        ]),
    html.Div(id = 'toprow',children=[
        dcc.Store(id='params_dict', storage_type='memory',data = {}),
        dcc.Store(id='dataset_titles', storage_type='memory',data = {}),
        dcc.Store(id='data_metadata_dict', storage_type='memory',data = {}),
        dcc.Store(id='columns_dict', storage_type='memory', data={"wav":[app_data.wn_string_name],'std':[i for i in app_data.STANDARD_COLUMN_NAMES],'oc':[]}),
        #dcc.Store(id='columns_dict', storage_type='memory', data={"wav":{app_data.wn_string_name},'std':{i for i in app_data.STANDARD_COLUMN_NAMES},'oc':set()}), #more effecient but doesn't work with framework
        dcc.Store(id='pretrained_model_metadata_dict', storage_type='memory', data={}),
        dcc.Store(id='pretrained_value_dict', storage_type='memory', data={'value':None}),
        dcc.Store(id='approaches_value_dict', storage_type='memory', data={'value': None}),
        dcc.Store(id='run_id', storage_type='memory'),
            html.Div(id='left col top row',children=[
            html.Div(id='datasets',children=[
                html.H2(id='H2_1', children='Select Datasets',
                    style={'textAlign': 'center','marginBelow':H2_height_below_padding, 'height':H2_height}),  #'marginTop': 20}
                dcc.Checklist(id='dataset-select',
                    options=get_datasets(), # [9:]if f"datasets/" == i[:8]
                    value=[], style={'maxHeight':200,'overflowY':'auto','width':left_body_width},inputStyle={"margin-right":checklist_pixel_padding_between }),
                dcc.Upload(
                    id='upload-ds',
                    children=html.Button('Upload Dataset(s)'),
                    multiple=True)
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
                        html.Div(id = "pretrained-holder")], style={'vertical-align': 'top', 'textAlign': 'center','maxHeight': 300,"overflowY":'auto','width':left_body_width})])
                ],style={"display": "inline-block",'marginRight': horizontal_pane_margin,'height': top_row_max_height, 'width': left_col_width}),

        html.Div(
            [
                html.H2(children='Data Columns',
                        style={'textAlign': 'center','height':H2_height,'marginBelow':H2_height_below_padding}),
                html.Div(id="data-pane",style={'height':top_row_max_height-H2_height-5,'maxHeight':top_row_max_height-H2_height-5,'overflowY':'auto'}),
            ], style={"display": "inline-block", 'vertical-align': 'top', 'textAlign': 'left','marginRight': horizontal_pane_margin,'width': middle_col_width,'height':top_row_max_height,'maxHeight':top_row_max_height}), #,

        html.Div(
            [
                html.H2(id='H2_3', children='Select Parameters',
                    style={'textAlign': 'center','marginBelow':H2_height_below_padding, 'height':H2_height}), #,'marginTop': 20
                html.Div(id = "params-holder"),
            ],style ={"display": "inline-block",'vertical-align': 'top','textAlign': 'left','width':right_col_width}),
        ],style={'height':top_row_max_height}),html.Hr(), #style={'marginBottom': 60}
    html.Div(id='middle_row',children=[

        html.Div(
            [
                html.Button('RUN',id="run-button",style=BUTTON_DEFAULT_STYLE),
                dcc.Loading(id='run-message',children=[])
            ], style={'textAlign': 'center','vertical-align': 'top'}),html.Hr(style={'marginBottom': 40})]), #,'marginLeft': 650
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
    Output('pretrained-holder', 'children', allow_duplicate=True),
    Input('mode-select', 'value'),
    Input('alert-model-success', 'is_open'),
    State('pretrained_value_dict', 'data'),
    prevent_initial_call = True
)
def update_pretrained_checklist(mode,is_open,pretrained_value):

    if mode != "Training":
        return [html.H4("Pretrained models:", style={'textAlign': "left"}),
dcc.Dropdown(id='pretrained-select', style={'width': 200}, options=get_pretrained(),value=pretrained_value['value']),
html.Div(id="pretrained-present", style={'textAlign': 'left'}),
 dcc.Upload(id='upload-pretrained', children=html.Button('Upload Pretrained Model(s)'), multiple=True, style={'textAlign': 'left'})
 ]
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

    pretrain_val = pretrain_val["value"]

    if mode != "Inference":
        opts = app_data.TRAINING_APPROACHES
        if mode == "Fine-tuning":
            if pretrain_dict != {}:
                #if/when we add this, make sure we are correctly converting compat_opts value to list
                if 'compatible_approaches' in pretrain_dict[pretrain_val]:
                    compat_opts = pretrain_dict[pretrain_val]['compatible_approaches']
                    opts = {key:val for (key,val) in opts.items() if key in compat_opts}
            #print(opts)
            opts = {key:val for (key,val) in opts.items() if opts[key]['finetunable']}

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

        approach_metadata = app_data.TRAINING_APPROACHES[approach]

        obj = [html.Div(children=[html.Strong(str(key)), html.Span(f": {approach_metadata[key]}")], style={'marginBottom': 10}) for key in approach_metadata if key not in hidden_metadata_keys]


        return [html.Div(children=obj)] #html.H5("Approach Info:"),
    else:
        return None

@callback(
    Output('pretrained-present', 'children'),
    Input('pretrained_model_metadata_dict', "data"),
    Input("mode-select", "value"),
    State("pretrained_value_dict", "data")
)
def present_pretrained_metadata(pretrained_model_metadata_dict,mode,pretrained):

    print("in present_pretrained_metadata")

    pretrained = pretrained['value']

    if mode != "Training":

        if pretrained!= None:
            pretrained_metadata = pretrained_model_metadata_dict[pretrained]
            obj = [html.Div(children=[html.Strong(str(key)), html.Span(f": {pretrained_metadata[key]}")],
                            style={'marginBottom': 10}) for key in pretrained_metadata]
            return [html.Div(children=obj)] #html.H5("Pretrained Info:"),
        else:
            return None

    else:
        return None

@callback(
    Output('data-pane',"children"),
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
    approach_val = approach_val['value']
    #what do we need to know for columns for model run?
    #training:
    #1. what standard columns and other columns are being used (and their order)
    #inference:
    #1. what standard columns and other columns can be used, as well as what's missing
    #fine tuning:
    #1 and #2 from above

    #print(datasets)
    #print(data_dict)
    #print(model_dict)

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
    valid_waves= 0
    #print(data_dict)
    for i in datasets:
        if data_dict[i]['wave_number_end_index']!="['-1']":
            #print(data_dict[i]['wave_number_end_index'])
            wave_counts.append(f"{int(ast.literal_eval(data_dict[i]['wave_number_end_index'])[0])-int(ast.literal_eval(data_dict[i]['wave_number_start_index'])[0])}" + \
                               f";{round(float(ast.literal_eval(data_dict[i]['wave_number_min'])[0]),2)};{round(float(ast.literal_eval(data_dict[i]['wave_number_max'])[0]),2)};" +\
                               f"{round(float(ast.literal_eval(data_dict[i]['wave_number_step'])[0]),2)}")
            valid_waves = valid_waves + 1

    wave_counts = set(wave_counts)

    #for training and inference, do not allow training for partial presence of wav numbers
    wav_str = f"{app_data.wn_string_name} valid: {valid_waves}/{ds_count}, equivalent: {len(wave_counts)}/{ds_count}"

    standard_cols_counter = Counter([i for sublist in [ast.literal_eval(data_dict[i]['standard_columns']) for i in datasets] for i in sublist])
    other_cols_counter = Counter([i for sublist in [ast.literal_eval(data_dict[i]['other_columns']) for i in datasets] for i in sublist])

    non_bio_columns = ['id','split'] #pipe in split to this, if specified in to be created and linked training parameters
    for i in non_bio_columns:
        if i in standard_cols_counter:
            standard_excluded.append((str(i),f'{i} ({standard_cols_counter[i]}/{ds_count}) (not a biological factor column)',standard_cols_counter[i]/ds_count))
            del standard_cols_counter[i]

    response_columns = ['age'] #pipe in split to this, if specified in to be created and linked training parameters
    for i in response_columns:
        if i in standard_cols_counter:
            standard_excluded.append((str(i),f'{i} ({standard_cols_counter[i]}/{ds_count}) (response column)',standard_cols_counter[i]/ds_count))
            del standard_cols_counter[i]

    wavs_exclude = False
    if pretrained_val != None:
        if mode == "Inference":
            if 'column_names' in model_dict[pretrained_val]:
                pretrained_include = [i for i in model_dict[pretrained_val]['column_names']] #pipe in split to this, if specified in to be created and linked training parameters
                #exclude wavs
                if app_data.wn_string_name not in pretrained_include:
                    wavs_exclude = True
                    wav_str = wav_str + " (not used in pretrained model)"
                #exclude standard
                pretrained_exclude = [i for i in standard_cols_counter if i not in pretrained_include]
                for i in pretrained_exclude:
                    standard_excluded.append((str(i),f'{i} ({standard_cols_counter[i]}/{ds_count}) (not used in pretrained model)',standard_cols_counter[i]/ds_count))
                    del standard_cols_counter[i]

                #exclude other cols
                pretrained_exclude = [i for i in other_cols_counter if i not in pretrained_include]
                for i in pretrained_exclude:
                    other_excluded.append((str(i),f'{i} ({other_cols_counter[i]}/{ds_count}) (not used in pretrained model)',other_cols_counter[i]/ds_count))
                    del other_cols_counter[i]
        elif mode == "Fine-tuning":
            pass
            #TODO: for already existing columns, mandate/suggest their inclusion.

            #print(standard_excluded)

    #filter out id from standard columns display, where it never should be used in training.
    standard_cols_counts_display = [(str(x[0]),f"{x[0]} ({x[1]}/{ds_count})",x[1]/ds_count) for x in sorted(standard_cols_counter.items(), key = lambda x: x[1], reverse = True)]
    other_cols_counts_display = [(str(x[0]),f"{x[0]} ({x[1]}/{ds_count})",x[1]/ds_count) for x in sorted(other_cols_counter.items(), key = lambda x: x[1], reverse = True)]

    wav_opts = [{"value":app_data.wn_string_name, "label":wav_str,"disabled":True if (valid_waves != ds_count or wavs_exclude) else False,
                 'extra':(valid_waves/ds_count,len(wave_counts)/ds_count)}]

    #consolidate all values, and apply previous selections (as relevant)

    wav_val = [] if (valid_waves != ds_count or wavs_exclude) else previous_selections['wav'] #previous_selections['wav']

    std_opts = [{"value":x[0],"label":x[1],"disabled":False,'extra':x[2]} for x in standard_cols_counts_display]+[{"value":x[0],"label":x[1],"disabled":True,'extra':x[2]} for x in standard_excluded]
    std_val = [m for m in [x[0] for x in standard_cols_counts_display] if m in previous_selections['std']]
    oc_opts = [{"value":x[0],"label":x[1],"disabled":False,'extra':x[2]} for x in other_cols_counts_display]+[{"value":x[0],"label":x[1],"disabled":True,'extra':x[2]} for x in other_excluded]
    oc_val = [m for m in [x[0] for x in other_cols_counts_display] if m in previous_selections['oc']]


    children = [html.Div(children=[html.H4("Wave numbers:"),dcc.Checklist(id='data-pane-wav-numbers',
                  options=wav_opts,  # [9:]if f"datasets/" == i[:8]
                  value=wav_val,inputStyle={"margin-right":checklist_pixel_padding_between})] if ds_count != 0 else None), #if len(standard_cols_counts_display)>0 or len(other_cols_counts_display)>0 else None
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

    print("in here"
    )

    print(selected_pretrained)

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

#possibly, could be issue with this where a dataset/model is reuploaded. To address this,
#could make sure to clear the item during upload process, or, disallow overwritting.
#there is an edge cases where a dataset could be uploaded through another mechanism, and match an exising name.
#however, this will be solved by a refresh which should seem like a logical first troubleshoot to the user
@callback(
    Output('data_metadata_dict',"data", allow_duplicate=True),
    Input('dataset-select','options'),
    Input('dataset-select','value'),
    State('data_metadata_dict',"data"),
    prevent_initial_call=True
)
def update_data_metadata_dict(known_datasets,selected_datasets,data_metadata_dict):

    kd_set = set(known_datasets)
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

    return data_metadata_dict


@callback(
    Output('download-results', "data"),
    Input('btn-download-results', 'n_clicks'),
    State('run-name','value'),
    State('mode-select', 'value'),
    State('run_id', 'data'),
    State('dataset_titles', 'data'),
    State('params_dict', 'data')
)
def download_results(n_clicks,run_name,run_id,dataset_titles,params_dict):

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

    print(prev)

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
            if 'options' in children:
                for p in children['options']:
                    if isinstance(p,dict):
                        p = p['value']
                    if p in children['value']:
                        out_dict[p] = True
                    else:
                        out_dict[p] = False
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
     State('approaches_value_dict', 'data'),
     State('data-pane', 'children'),
     State('dataset-select', 'value'),
     State('params-holder', 'children'),
     State("dataset_titles", "data"),
    prevent_initial_call=True
 )
def model_run_event(n_clicks,mode,pretrained_model,approach,columns,datasets,params_holder,dataset_titles):
    pretrained_model = pretrained_model['value']
    approach = approach['value']

    print(pretrained_model)
    print(approach)
    #loop through and look for "values". if exists options, use values to determine true or false.
    #Use id as the parameter name, make sure that convention is kept in get_params

    #this object enables us to use the 'selected' and 'extra' (for wavs, (%valid,%equivalent) and for others % present in total ds) and assess whether it is fine, warning, or error
    data_pane_vals_dict = unpack_data_pane_values_extra({}, columns)

    print(data_pane_vals_dict)

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
                data,artifacts,stats,dataset_titles = run_model(mode,approach,datasets,run_id)

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
def test_fxn2(slider_val):

   return f"Train (%): {slider_val[0]}",f"Validation (%): {slider_val[1]-slider_val[0]}",f"Test (%): {100-slider_val[1]}"
@callback(Output('splits_status',"children"),
                Output('splits_choice',"children"),
                Input('define-splits', 'value')
          )
def test_fxn(splits_val):

    #return f"Define splits: {splits_val}",[html.H5("| -- Train -- | -- Val -- | -- Test -- |"),dcc.RangeSlider(0, 100, 5, value=[60, 80], id='splits-slider', allowCross=False),html.Div("0% to the first value is the training split, the first value to the second value is the validation split, and the second value to 100% is test. Numbers represent percentages. Train and val must each be > 0%"),] if splits_val else None

    return f"Define splits: {splits_val}", [html.Div(id='train_val'),html.Div(id='val_val'),html.Div(id='test_val'),
                                            dcc.RangeSlider(0, 100, 1, marks = {x*5:x*5 for x in range(20)}, value=[60, 80], id='splits-slider',
                                                            allowCross=False)]  if splits_val else None
@callback(#Output('params-select', 'options'),
                    #Output('params-select', 'value'),
                    Output('params-holder', 'children'),
                    Input('mode-select', 'value'),
                    Input('approaches_value_dict', 'data')
)
def get_parameters(mode,approach):
    approach = approach['value']

    params_holder_subcomponents = []

    if mode == 'Training' or mode == 'Fine-tuning':

            params_holder_subcomponents.append(html.Div(id='Training_params',children=[
                html.H4("Training"),
                html.Div(id='splits_status',children='Define splits: False'),
                daq.ToggleSwitch(id='define-splits',value=False),
                html.Div(id='splits_choice')
            ]))

            #training
            #this will largely be populated manually (hardcoded).  Right now, that will look like Michael explaining to Dan
            #the relevant training hyperparameters to expose per approach ("archetecture").
            if approach is not None:
                if approach == "Basic model":
                    #return ["test"],[],dcc.Slider(0, 20, 5,value=10,id='test-conditional-component')
                    params_holder_subcomponents.append(html.Div(id='model_params',children=[html.H4("Basic model"),dcc.Checklist(id='checklist-params', options=["test"], value=[],inputStyle={"margin-right":checklist_pixel_padding_between}),
                            html.Div(id='a_specific_param-name',style={'textAlign': 'left'},children="a_specific_param"),
                            dcc.Slider(0, 20, 5,value=10,id='a_specific_param')]))
                elif approach == "hyperband tuning model":
                    params_holder_subcomponents.append(html.Div(id='model_params',children=[html.H4('hyperband tuning model'),dcc.Checklist(id='checklist-params', options=["on_GPU","other thing"], value=[],inputStyle={"margin-right":checklist_pixel_padding_between})]))
                elif approach == "new_exp_arch":
                    #return ["on_GPU","other thing","secret third thing"],[],[]
                    params_holder_subcomponents.append(html.Div(id='model_params',children=[html.H4('new_exp_arch'),dcc.Checklist(id='checklist-params', options=["on_GPU","other thing","secret third thing"], value=[])]))
    elif mode == 'Inference':

            #use the current selection and model metadata dict to locate any model specific inference settings..?

            params_holder_subcomponents.append(html.Div(id = 'Inference_params',children=[
                html.H4("Inference"),dcc.Checklist(id='checklist-params', options=["on_GPU"], value=[False],inputStyle={"margin-right":checklist_pixel_padding_between})
            ]))

    else:
        return None

    return params_holder_subcomponents



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

                blob = STORAGE_CLIENT.bucket(DATA_BUCKET).blob(f'datasets/{i[0]}')

                blob.upload_from_string(data, 'text/csv')

                attach_metadata_to_blob(metadata,blob)

                #if filename is already in data_metadata_dict, clear it so that the contents will be properly
                #pulled


                if i[0] in data_metadata_dict:
                    print('clearing data_metadata_dict on reupload')
                    del data_metadata_dict[i[0]]

                message = ""
                #import code
                #code.interact(local=dict(globals(), **locals()))

            else:
                valid = False
    else:
        return data_metadata_dict,None,None,None,None,get_datasets()

    return data_metadata_dict,valid,not valid,message,None,get_datasets()

#similar to the one below,

@callback(
    Output("model-upload-from-session-success", "is_open"),
    Output("model-upload-from-session-failure", "is_open"),
    Input('btn-upload-model', 'n_clicks'),
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
def models_to_gcp(filename, contents):

    success = True
    # this gets called by alert on startup as this is an output, make sure to not trigger alert in this case
    if not filename == None and not contents == None:
        for i in zip(filename, contents):

            valid,message,data,metadata = data_check_models(i,load_model = True)

            if valid:

                blob = STORAGE_CLIENT.bucket(DATA_BUCKET).blob(f'models/{i[0]}')

                blob.upload_from_string(data, 'text/csv')

                #for every piece of metadata, attach it to the cloud object as well for easy retrieval
                if metadata is not None:
                    print('attaching metadata')
                    attach_metadata_to_blob(metadata, blob)
                else:
                    attach_null_metadata(blob)

            else:
                success = False

        return success, not success, message
    else:
        return False, False, None
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

        # check that the dataset does not contain duplicated columns:

        print('checking duplicate column names')

        # avoid behavior of pandas to rename duplicate names
        reader = csv.reader(io.StringIO(decoded.decode('utf-8')))
        columns = next(reader)

        # check that column names are unique
        if len(columns) != len(set(columns)):
            valid = False
            message = message + "; - " + f"columns must be uniquely named"

        print("loaded")

        # along with parsing columns, rename the column metadata to 'standard' names. For example,
        # the identifer field, no matter what it is called, will be replaced by id. Have an visual indicator
        # for columns that cannot be matched to global naming.

        # what I could do here- have a google sheet that matches global names to variable type, as well
        # as known aliases. This can be referenced here, as well as for model behaviors like one-hot expansion
        # of categorical variables.

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

            other_cols_start = 0  # assume this for now, but see if there are exeptions
            other_cols_end = wave_number_start_index - 1

            non_wave_columns = columns[other_cols_start:other_cols_end]
        else:
            non_wave_columns = columns
            wave_number_start_index = "-1"
            wave_number_end_index = "-1"
            wave_number_min = "-1"
            wave_number_max = "-1"
            wave_number_step = "-1"

        # classify provided columns into standard and other columns:

        standard_cols = []
        standard_cols_aliases = []
        other_cols = []

        for f in non_wave_columns:
            if f in app_data.STANDARD_COLUMN_NAMES:
                if f not in standard_cols:
                    standard_cols.append(f)
                    standard_cols_aliases.append(f)
                else:
                    valid = False
                    message = message + "; - " + f"multiple columns refer to the same internal id: {f}"
            else:
                # match = [f in x[0] for x in [(standard_variable_names.DICT[p]['aliases'],p) for p in standard_variable_names.DICT]]
                match = [p for p in app_data.STANDARD_COLUMN_NAMES if f in app_data.STANDARD_COLUMN_NAMES[p]['aliases']]
                if len(match) > 1:
                    valid = False
                    message = message + "; - " + f"column name {f} matched aliases for multiple standard names: issues is a duplicate alias in standard names dictionary, remedy there."
                elif len(match) > 0:
                    if match[0] not in standard_cols:
                        standard_cols.append(match[0])
                        standard_cols_aliases.append(f)
                    else:
                        valid = False
                        message = message + "; - " + f"multiple columns refer to the same internal id: {match[0]}"
                else:
                    other_cols.append(f)

        # If wav numbers are absent in dataset, supply relevant metadata fields with -1
        metadata = {"standard_columns": standard_cols, "standard_column_aliases": standard_cols_aliases,
                    "other_columns": other_cols, "wave_number_start_index": [wave_number_start_index],
                    "wave_number_end_index": [wave_number_end_index], "wave_number_min": [wave_number_min],
                    "wave_number_max": [wave_number_max], "wave_number_step": [wave_number_step]}
        if valid:
            return True,"",decoded if load_data else None,metadata #
        else:
            return False,message,None,None

def data_check_models(file,load_model):

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

#this is the  actual ML integration etc. placeholder

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

