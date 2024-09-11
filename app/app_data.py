#reflect the decisions made here:
#as long as above is a up to date document.

#alternatively, we could make this a google sheet for a little easier editing and fine grained
#permissions.

#placed in a seperate file to discourage accidental changes from editors of this to app.py

#can not have alias duplicated between standard names. In this case, make one scientist change their naming.

WN_STRING_NAME = 'wn**'

#https://docs.google.com/document/d/1OfonMVbnT2u4JrQZMdYMjhif2y8EDYa8MGdn1JW9K9w/edit
#STANDARD_COLUMN_NAMES = {'File_name':{'data_type':'unq_text'},
#        'Sample':{'data_type':'int'},
#        #'catch time':{'data_type':'int'}, slightly different than specified, assuming we will break this up ultimately
#        'Catch year' : {'data_type':'int'},
#        'Catch month': {'data_type': 'int'},
#        'Catch day': {'data_type': 'int'},
#        'Sex' : {'data_type':'categorical'},
#        'Final_age' : {'data_type':'numeric'},
#        'Latitude' : {'data_type':'numeric'},
#        'Longitude' : {'data_type':'numeric'},
#        'Region': {'data_type': 'categorical'},
#        'Length': {'data_type': 'numeric'},
#        #'fish_weight': {'data_type': 'numeric'},
#        'Otolith weight': {'data_type': 'categorical'},
#        'Gear_depth': {'data_type': 'numeric'},
#        'Gear_temp': {'data_type': 'numeric'}
#        }

#NON_BIO_COLUMNS = ['File_name','Sample']
#RESPONSE_COLUMNS = ['Final_age']

idname = "id"
splitname = "split"
responsename = "age"

STANDARD_COLUMN_NAMES = {idname:{'data_type':'unq_text'},
        splitname:{'data_type':'int'},
        'catch_year' : {'data_type':'int'},
        'catch_month': {'data_type': 'int'},
        'catch_day': {'data_type': 'int'},
        'sex' : {'data_type':'categorical'},
        responsename : {'data_type':'numeric'},
        'latitude' : {'data_type':'numeric'},
        'longitude' : {'data_type':'numeric'},
        'region': {'data_type': 'categorical'},
        'fish_length': {'data_type': 'numeric'},
        'fish_weight': {'data_type': 'numeric'},
        'otolith_weight': {'data_type': 'categorical'},
        'gear_depth': {'data_type': 'numeric'},
        'gear_temp': {'data_type': 'numeric'}
        }

NON_BIO_COLUMNS = [idname,splitname]
RESPONSE_COLUMNS = [responsename]

#currently, just hardcoding parameters as doing this dynamically seems anti-pattern with the framework, and
#sometimes parameters will need to have interactive effects which needs to be coded in dash.
#to update the parameters for each approach, hardcode in get_parameters() in app.py
#'mandatory_cols':[wn_string_name],'excluded_cols':[]
TRAINING_APPROACHES = {"Basic model":{'description':"basic, customizable model",'finetunable':True},
                       "hyperband tuning model":{'description':"A version of the basic model with hyperband parameter tuning",'finetunable':False}}
