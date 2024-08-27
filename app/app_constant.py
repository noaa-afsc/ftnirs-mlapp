from dash import html,dcc,Input,Output
import os

first_row_height = 10
second_row_height = 15
second_row_padding = 5
header_height = second_row_height + second_row_padding + first_row_height
#used in all pages
app_header = html.Div([
        dcc.Interval(
        id='five-second-interval',
        interval=5*1000,  # in milliseconds (5 seconds)
        n_intervals=0  # start at zero
    ),html.Div(
        children=[
        html.Div(id="topline", children=[
                html.Div(html.H1('Fourier Transform Near-Infrared Spectroscopy Machine Learning tool',
                style={"font-size":"160%","paddingBelow":0,'margin': 0}),style={"display": "inline-block"}),
            #and then remove on stop

        #html.Div(id="secondline", children = [
                dcc.Link(html.Img(src="../static/home.png",style={"width":"20px","marginLeft":10,"marginRight":10}),
                     href=f"/{os.getenv('APPNAME')}/",style={"display": "inline-block","margin":0}),
                dcc.Link(html.Img(src="../static/question-sign.png",style={"width":"20px","marginRight":10}),
                        href=f"/{os.getenv('APPNAME')}/help",style={"display": "inline-block"}),
                html.Div("Number current running jobs:", style={"display": "inline-block", 'textAlign': 'right'}),
                # "float": "right"
                html.Div(id="active-jobs-holder", style={"display": "inline-block"}),
                # make the job run fxn increment a global var for this
                html.Div([
                    html.Div(f"Web tool Github release version: {os.getenv('WEBAPP_RELEASE')}",
                             style={"display": "inline-block", 'textAlign': 'right'}),
                    html.A(
                        html.Img(src="../static/github-sign.png", style={"width": "20px", "marginLeft": 10, "marginRight": 10}),
                        href="https://github.com/DanWoodrichNOAA/ftnirs-mlapp/tree/ml-dev", style={"display": "inline-block", "marginRight": 15}, target="_blank"),
                    html.Div(f"ML-codebase Github release version: {os.getenv('MLCODE_RELEASE')}",
                             style={"display": "inline-block", 'textAlign': 'right'}),
                    html.A(
                        html.Img(src="../static/github-sign.png", style={"width": "20px", "marginLeft": 10, "marginRight": 10}),
                        href="https://github.com/michael-zakariaie-noaa/ftnirs-ml-codebase/tree/dan-dev", style={"display": "inline-block", "margin": 0}, target="_blank")
                ],style={"float": "right"})],style={"height": first_row_height,"display": "inline-block","width":"100%","backgroundColor":"white"})],
            #    ],style={"float":"right"})],
            style={'margin': 0,"vertical-align":"middle"}),
        html.Div(style={"height":second_row_height,"paddingBelow":second_row_padding,'margin': 0,"backgroundColor":"white"}),
        html.Hr(style={"paddingAbove":0,'margin': 0,"paddingBelow":0})
        ],style={"position":"fixed","width":"100%","zIndex":1000,"backgroundColor":"white",'height':header_height})

#maybe some values to add in to the header alongside current running jobs.

#!/usr/bin/env python
#import psutil
# gives a single float value
#psutil.cpu_percent()
# gives an object with many fields
#psutil.virtual_memory()
# you can convert that object to a dictionary
#dict(psutil.virtual_memory()._asdict())
# you can have the percentage of used RAM
#psutil.virtual_memory().percent
#79.2
# you can calculate percentage of available memory
#psutil.virtual_memory().available * 100 / psutil.virtual_memory().total
#20.8