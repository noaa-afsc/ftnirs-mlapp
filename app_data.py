#reflect the decisions made here:
#as long as above is a up to date document.

#alternatively, we could make this a google sheet for a little easier editing and fine grained
#permissions.

#placed in a seperate file to discourage accidental changes from editors of this to app.py

#can not have alias duplicated between standard names. In this case, make one scientist change their naming.

wn_string_name = 'wn**'

STANDARD_COLUMN_NAMES = {'id':{'data_type':'unq_text','aliases':['File_name','file_name']},
        'split':{'data_type':'int','aliases':['Sample','sample']},
        'catch_time':{'data_type':'int','aliases':[]},
        'sex':{'data_type':'categorical','aliases':[]},
        'age':{'data_type':'numeric','aliases':['Final_age','final_age']},
        'latitude':{'data_type':'numeric','aliases':['Latitude']},
        'longitude':{'data_type':'numeric','aliases':['Longitude']},
        'region': {'data_type': 'categorical', 'aliases': ['Region']},
        'fish_length': {'data_type': 'numeric', 'aliases': ['Length','length']},
        'fish_weight': {'data_type': 'numeric', 'aliases': ['weight']},
        'otolith_weight': {'data_type': 'categorical', 'aliases': ['Otolith weight']},
        'gear_depth': {'data_type': 'numeric', 'aliases': ['Gear_depth']},
        'gear_temp': {'data_type': 'numeric', 'aliases': ['Gear_temp']},
        }

#currently, just hardcoding parameters as doing this dynamically seems anti-pattern with the framework, and
#sometimes parameters will need to have interactive effects which needs to be coded in dash.
#to update the parameters for each approach, hardcode in get_parameters() in app.py
TRAINING_APPROACHES = {"michael_deeper_arch":{'description':"a new approach with a deeper architecture",'mandatory_cols':[wn_string_name],'excluded_cols':[]},
                       "irina_og_arch":{'description':"original approach",'mandatory_cols':[wn_string_name],'excluded_cols':[]},
                       "new_exp_arch":{'description':"an approach that does not use NN",'mandatory_cols':[wn_string_name],'excluded_cols':[]},
                       "wacky_arch":{'description':"an approach that excludes wave numbers",'mandatory_cols':[],'excluded_cols':[wn_string_name]}}