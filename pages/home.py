import csv
import os
import tarfile
import io
import dash
import dash_daq as daq
import plotly.graph_objects as go
from dash import dcc,html,State, dash_table,callback_context, callback
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import pandas as pd
import random as rand
from google.cloud import storage
import plotly.graph_objects as go
import time
import base64
import h5py
import app_data
from dotenv import load_dotenv

load_dotenv('./tmp/.env')

dash.register_page(__name__, path='/',name = os.getenv("APPNAME")) #

#this should be an .env variable
GCP_PROJ="ggn-nmfs-afscftnirs-dev-97fc"

#STORAGE_CLIENT = storage.Client(project=os.getenv("GCP_PROJ"))
STORAGE_CLIENT = storage.Client(project=GCP_PROJ)
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
    return [i for i in app_data.TRAINING_APPROACHES]

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
    html.Div(
        [dcc.Store(id='params_dicts', storage_type='memory',data = {"params_dict":{},"params_dict_run":{}}),
        dcc.Store(id='dataset_titles', storage_type='memory',data = {}),
        dcc.Store(id='data_metadata_dict', storage_type='memory',data = {}),
        dcc.Store(id='model_metadata_dict', storage_type='memory', data={}),
        dcc.Store(id='run_id', storage_type='memory'),
        html.Div(
    [
                html.H2(id='H2_1', children='Select Datasets',
                    style={'textAlign': 'left', 'marginTop': 20}),
                dcc.Checklist(id='dataset-select',
                    options=get_datasets(), # [9:]if f"datasets/" == i[:8]
                    value=[],style={'width':800}),
                dcc.Upload(
                    id='upload-ds',
                    children=html.Button('Upload File(s)'),
                    multiple=True)
            ],style={"display": "inline-block",'vertical-align': 'top','textAlign': 'left','marginRight': 20}),

        html.Div(
    [
                html.H2(id='H2_2', children='Select Mode',
                    style={'textAlign': 'center','marginTop': 20,'textAlign': 'left'}),
                dcc.Dropdown(id='mode-select', value="Training",clearable=False,
                     options=["Training", "Inference", "Fine-tuning"],style={'width':200}),
                html.Div(id='mode-select-output',style = {'textAlign': 'left'}),
                dcc.Dropdown(id='model-select',style={'width':200}),
                html.Div(id='model-select-output',style={'textAlign': 'left'}),
                dcc.Upload(id='upload-model',children=None,multiple=True,style={'textAlign': 'left'})
            ],style ={"display": "inline-block",'vertical-align': 'top','textAlign': 'center','width': 300,'marginRight':100,'height':450}),

        html.Div(
            [
                html.H2(id='H2_3', children='Select Parameters',
                    style={'textAlign': 'center','marginTop': 20}),
                html.Div(id = "params-holder"),
            ],style ={"display": "inline-block",'vertical-align': 'top','textAlign': 'right'}),
        ]),html.Hr(style={'marginBottom': 40}),
    html.Div(id='middle_row',children=[

        html.Div(
            [
                html.H2(children='Data Columns',
                        style={'textAlign': 'left'}),
                html.Div(id="data-pane"),
            ], style={"display": "inline-block", 'vertical-align': 'top', 'textAlign': 'left','width': 800,'marginRight': 200}),

        html.Div(
            [
                html.Button('RUN',id="run-button"),
                dcc.Loading(id='run-message',children=[])
            ], style={'textAlign': 'center','vertical-align': 'top',"display": "inline-block"})]), #,'marginLeft': 650
html.Div(
        [
        html.Div(
        [
                    html.Div(id='config-report',children =[],style={'textAlign': 'left','vertical-align': 'top','width': 580,'height': 300})#
             ],style={"display": "inline-block",'vertical-align': 'top','textAlign': 'center','marginRight': 200}),
        html.Div(
            [
                    html.Div(id="stats-out"),
                    html.Div(id = "artifacts-out")
            ],style={"display": "inline-block",'vertical-align': 'top','textAlign': 'center','marginRight': 200}),
        html.Div(
                [

                    html.Div(id = "download-out"),
                    html.Div(id = "upload-out")
                ],style={"display": "inline-block",'vertical-align': 'top'})
        ],style={"display": "inline-block",'vertical-align': 'top'})

])

@callback(
    Output('model_metadata_dict',"data", allow_duplicate=True),
    Input('mode-select', 'value'),
    Input('model-select','options'),
    Input('model-select','value'),
    State('model_metadata_dict',"data"),
    prevent_initial_call=True
)
def update_model_metadata_dict(mode,known_models,selected_models,model_metadata_dict):

    if mode == "Training" or selected_models == None:
        return model_metadata_dict
    else:

        selected_models = [selected_models]

        km_set = set(known_models)
        mmd_set = set(model_metadata_dict)

        excess_metadata = mmd_set.difference(km_set)

        if len(excess_metadata) > 0:
            # remove out of date metadata
            for k in excess_metadata:
                print("deleting "+k)
                del model_metadata_dict[k]

        uk_models = [k for k in selected_models if k not in mmd_set]
        for k in uk_models:
            print(k)
            blob = STORAGE_CLIENT.bucket(DATA_BUCKET).get_blob(f'models/{k}')
            if blob == None:
                print("model does not exist in cloud storage - list not refreshed. bug?")
            elif blob.metadata != None:
                model_metadata_dict.update({k: blob.metadata})
            else:
                data_bytes = blob.download_as_bytes()
                iobytes = io.BytesIO(data_bytes)

                valid,_,data,metadata = data_check_models((k, "data:application/octet-stream;base64," + base64.b64encode(data_bytes).decode('utf-8')),load_model=True)

                if not valid:
                    metadata = {"no_metadata":1}
                    # flash a warning, but store in the metadata an indicator that the ds is not eligible

                # upload the flag to
                print(metadata)
                model_metadata_dict.update({k:metadata})
                attach_metadata_to_blob(metadata, blob)


    print(len(model_metadata_dict))
    print(model_metadata_dict)

    return model_metadata_dict



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
            data_metadata_dict.update(metadata)
            attach_metadata_to_blob(metadata,blob)

    print(len(data_metadata_dict))
    print(data_metadata_dict)

    return data_metadata_dict

    #add feature: if a selected dataset is missing metadata, need to load in, submit, and register
    #from within this function

#@callback(
#    Output('data-pane','options'),
#    Input('dataset-select','value'),
#    Input('model-select', 'value'),
#    State('mode-select', 'value'),
#)

#def populate_data_pane(datasets):

    out_dict = {}

#    for i in datasets:
#        blob = STORAGE_CLIENT.bucket(DATA_BUCKET).get_blob(f'datasets/{model}')

#        metadata = blob.metadata



@callback(
    Output('download-results', "data"),
    Input('btn-download-results', 'n_clicks'),
    State('run-name','value'),
    State('mode-select', 'value'),
    State('run_id', 'data'),
    State('dataset_titles', 'data')
)
def download_results(n_clicks,run_name,mode,run_id,dataset_titles):

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
            if mode !="Inference":
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
     Output("params_dicts","data"),
     Output("dataset_titles", "data"),
     Output("run_id", "data"),
     Input('run-button', 'n_clicks'),
     State('mode-select', 'value'),
     State('model-select', 'value'),
     State('dataset-select', 'value'),
     State('params_dicts', 'data'),
     State("dataset_titles", "data"),
    prevent_initial_call=True
 )
def model_run_event(n_clicks,mode,model,datasets,params_dicts,dataset_titles):

    run_id = "" #set default so if it errors out, still can return outputs

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
                if params_dicts["params_dict_run"] == {}:
                    params_dicts["params_dict_run"] = params_dicts["params_dict"]

                # time.time() makes it unique even when parameters are fixed, may want to change behavior later
                run_id = str(abs(hash("".join(["".join(params_dicts["params_dict_run"]), str(mode), str(model), "".join(datasets)])+str(time.time()))))

                config_dict = params_dicts["params_dict_run"].copy()

                config_dict.update({"Datasets":",".join(datasets)})
                config_dict.update({"Model":model})
                config_dict.update({"Mode":mode})

                config_table = pd.DataFrame(config_dict,index=[0])

                blob = STORAGE_CLIENT.bucket(TMP_BUCKET).blob(f'config_{run_id}.yml')
                blob.upload_from_string(config_table.to_csv(), 'text/csv')

                params_dicts["params_dict_run"] = params_dicts["params_dict"]

                config_out_children = [html.Div(id='run-name-block',children =[html.Div(id='run-name-prompt',children = "Run name:"),
                                        dcc.Input(id='run-name', type="text", placeholder="my_unique_pretrained_model_name",style={'textAlign': 'left', 'vertical-align': 'top', 'width': 400}),
                                                                               html.Div(id='description-prompt', children="Description:"),
                                        dcc.Textarea(id='description',style={'textAlign': 'left','vertical-align': 'top','width': 400,'height':100})
                                                                                ],style={"display": "inline-block"}) if mode!="Inference" else html.Div(id='run-name-block',children=[html.Div(id='run-name'),html.Div(id='description')]),] +\
                                       [html.Div(id='config-report-rc',children = "Run Configuration:"),
                                       html.Div(id='config-report-mode', children="Mode: "+ mode),
                                       html.Div(id='config-report-model',children="Model: " + model),
                                       html.Div(id='config-report-datasets',children ='Datasets: ')] +\
                                       [html.Div(id='config-report-datasets-' + i,children="- "+str(i),style={'marginLeft': 15}) for i in datasets] +\
                                       [html.Div(id='config-report-parameters',children ='Parameters: ')] + \
                                       [html.Div(id='config-report-parameters-' + a,children="- "+a+": "+str(b),style={'marginLeft': 15}) for (a,b) in params_dicts["params_dict"].items()]

                #this is where model will actually run
                data,artifacts,stats,dataset_titles = run_model(mode,model,datasets,run_id)

                message = "Run Succeeded!"

                config_out_payload = config_out_children

            except Exception as e:
                message = "Run Failed: error while processing algorithm"
                config_out_payload = [html.Div(id='error-title',children="ERROR:"),html.Div(id='error message',children =[str(e)])]
                processing_fail = True


        download_out = [html.Div([html.Button("Download Results", id="btn-download-results"),
                    dcc.Download(id="download-results")]) if not (processing_fail or any_error) else ""]

        artifacts_out = html.Div(artifacts,style={'textAlign': 'left', 'vertical-align': 'top','width': 400})

        stats_out = html.Div([
                        html.Div(
                        [
                            html.Div(id='stats-title',children="Run stats:"),
                            dash_table.DataTable(stats.to_dict('records')),
                        ],style={'textAlign': 'left', 'vertical-align': 'top','width': 400}) if not (processing_fail or any_error) else ""])

        return [processing_fail,ds_fail,model_fail,message,config_out_payload,stats_out,artifacts_out,download_out,html.Button("Upload Trained model", id="btn-upload-model") if mode != "Inference" and not (processing_fail or any_error) else "",params_dicts,dataset_titles,run_id]

@callback(#Output('params-select', 'options'),
                    #Output('params-select', 'value'),
                    Output('params_dicts',"data", allow_duplicate=True),
                    Output('params-holder', 'children'),
                    Input('mode-select', 'value'),
                    Input('model-select', 'value'),
                    State("params_dicts","data"),
    prevent_initial_call=True
)
def get_parameters(mode,model,params_dicts):

    #if this is present, wipe it out
    params_dicts['params_dict'] = {}

    if model is not None:
        if mode == 'Inference':
            #we will ideally want this to read properties of the model object itself to run this.
            #hopefully will not be particular to a certain model.

            return params_dicts,dcc.Checklist(id='checklist-params', options=["on_GPU"], value=[False])

            #if model == "hello4.hd5":
            #    return [dcc.Checklist(id='checklist-params', options=["on_GPU","specific other thing","secret third thing"], value=[False,False,False])]
            #else:
            #    return []
        #training
        else:
            #this will largely be populated manually (hardcoded).  Right now, that will look like Michael explaining to Dan
            #the relevant training hyperparameters to expose per modeling approach ("archetecture").
            if model == "michael_deeper_arch":
                #return ["test"],[],dcc.Slider(0, 20, 5,value=10,id='test-conditional-component')
                return params_dicts,[dcc.Checklist(id='checklist-params', options=["test"], value=[]),
                        html.Div(id='var1-param-name',style={'textAlign': 'left'},children="a_specific_param"),
                        dcc.Slider(0, 20, 5,value=10,id='var1-param')]
            elif model == "irina_og_arch":
                #return ["on_GPU","other thing"],[],[]
                return params_dicts,[dcc.Checklist(id='checklist-params', options=["on_GPU","other thing"], value=[])]
            elif model == "new_exp_arch":
                #return ["on_GPU","other thing","secret third thing"],[],[]
                return params_dicts,[dcc.Checklist(id='checklist-params', options=["on_GPU","other thing","secret third thing"], value=[])]
    else:
        return params_dicts,[]


@callback(
    Output('model-select', 'options'),
    Output('upload-model', 'children'),
    Input('mode-select', 'value'),
    Input('alert-model-success', 'is_open')
)
def update_model_checklist(mode,is_open):

    if mode != "Training":
        return get_models(),html.Button('Upload Trained Model(s)')
    else:
        return get_archetectures(),None

@callback(
    Output('mode-select-output', 'children'),
    Input('mode-select', 'value')
)
def update_output(mode):

    if mode == "Training":
        definition = "Train a model from scratch using a predefined modeling approach"
    if mode == "Inference":
        definition = "Predict ages without needing age information"
    if mode == "Fine-tuning":
        definition = "Select a pretrained model and fine-tune it for better performance on smaller datasets"
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
    Output('model-select-output', 'children'),
    Input("model_metadata_dict","data"),
    State('model-select', 'value'),
    State('mode-select', 'value')
)
def present_metadata(model_metadata_dict,model,mode):

    print(model_metadata_dict)

    if model == None:
        return None

    if mode == 'Training':

        metadata = app_data.TRAINING_APPROACHES[model]

    else:
        #extract metadata from GCP object, if available (lightest option)

        metadata = model_metadata_dict[model]

    # could interpret this a little more: translate to 'return html.Div("File object missing metadata")'
    # Don't dump all the info, rather, weed out the metadata fields and present the info that is most valuable to the user
    return [html.Div(children=[html.Strong(str(key)),html.Span(f": {metadata[key]}")],style={'marginBottom': 10}) for key in metadata]

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

                print(i[0])

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
    Input('upload-model', 'filename'),Input('upload-model', 'contents')
)
def models_to_gcp(filename, contents):

    success = True
    # this gets called by alert on startup as this is an output, make sure to not trigger alert in this case
    if not filename == None and not contents == None:
        for i in zip(filename, contents):
            print(i[1][0:200])

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

        print(columns)

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
            standard_cols = ["-1"]
            standard_cols_aliases = ["-1"]
            wave_number_start_index = ["-1"]
            wave_number_end_index = ["-1"]
            wave_number_min = ["-1"]
            wave_number_max = ["-1"]
            wave_number_step = ["-1"]

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
                    "other_columns": other_cols, "wave_number_start_index": wave_number_start_index,
                    "wave_number_end_index": wave_number_end_index, "wave_number_min": wave_number_min,
                    "wave_number_max": wave_number_max, "wave_number_step": wave_number_step}
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


################################as add parameters, give it this boilerplate:

@callback(Output("params_dicts","data", allow_duplicate=True),
              Input('checklist-params', 'value'),
              Input('checklist-params', 'options'),
              #Input('run-button', 'n_clicks'),
              Input('params-holder', 'children'),
              State('params_dicts',"data"),
    prevent_initial_call=True
)
def checklist_params_dict_pop(value,options,_,params_dicts):

    print('in 2')

    out_dict = {i: (True if i in value else False) for i in options}
    for i in out_dict:
        params_dicts["params_dict"][i]= out_dict[i]

    return params_dicts

@callback(Output("params_dicts","data", allow_duplicate=True),
                Input('var1-param', 'value'),
                Input('var1-param-name', 'children'),
                Input('params-holder', 'children'),
                State('params_dicts', "data"),
    prevent_initial_call=True
)
def checklist_params_dict_pop_var1_param(value,name,_,params_dicts):

    print('in 1')

    params_dicts["params_dict"][name] = value

    return params_dicts

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

