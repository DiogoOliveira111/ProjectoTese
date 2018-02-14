import pickle
import dash
import dash_core_components as dcc
import plotly.graph_objs as go
import pandas as pd
from WBMTools.sandbox.interpolation import interpolate_data
from tkinter import Tk, Label
import matplotlib.pyplot as plt
import numpy as np
import plotly.plotly as py
import dash_html_components as html
import easygui

path = easygui.fileopenbox()

with open(path, 'rb') as handle:
    collection= pickle.load(handle)

flag=0
MouseTime=[]
MouseX=[]
MouseY=[]
for i in collection:

    event=collection[i]
    if( event['Type']=='Mouse'):
        data=event['Data'].split(';')
        if (i==0):
            initial_time = float(data[-1])
            MouseTime.append(initial_time/1000)
        else:
            MouseTime.append((float(data[-1]) - initial_time) / 1000)
        MouseX.append(float(data[2]))
        MouseY.append(float(data[3]))
        flag=1 #Flag to determine if there is Mouse data in the collection

    if(flag==0):
        root= Tk()

        # Make window 300x150 and place at position (50,50)
        root.geometry("600x300+50+50")

        # Create a label as a child of root window
        my_text = Label(root, text='The Collection chosen has no Mouse Data')
        my_text.pack()
        root.mainloop()
        exit()


MouseDict = dict(t=MouseTime, x=MouseX, y=MouseY)
dM = pd.DataFrame.from_dict(MouseDict)
# time_var,space_var=interpolate_data(dM,t_abandon=20)

app = dash.Dash()

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}


app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        children='Hello Dash',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),

    html.Div(children='Dash: A web application framework for Python.', style={
        'textAlign': 'center',
        'color': colors['text']
    }),

html.Div(children= dcc.Graph(
        id='PosGraf',
        figure={
            'data': [
                go.Scatter(
                    x = dM['x'],
                    y= dM['y'],
                    mode= 'markers',
                    opacity=0.7,
                    marker={
                        'size':7,
                        'line':{'width' : 0.5 , 'color': 'white'}
                    }
                )

            ],
            'layout': go.Layout(
                xaxis={ 'title': 'X position'},
                yaxis={'title': 'y position'},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                legend={'x': 0, 'y': 1},
                hovermode='closest')

        }

    ))
])

if __name__ == '__main__':
    app.run_server(debug=False)