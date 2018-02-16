import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
import easygui
import pickle
from tkinter import Tk, Label
from WBMTools.sandbox.interpolation import interpolate_data

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
time_var,space_var=interpolate_data(dM,t_abandon=20)

app = dash.Dash()

app.layout = html.Div([
    dcc.Graph(id='graph-with-slider'),
    dcc.RadioItems(
        id='Radio',
       options=[
           {'label': 'Velocity in X', 'value': 'vx'},
           {'label': 'Velocity in Y', 'value': 'vy'},
           {'label': 'Jerk', 'value': 'jerk'},
           {'label': 'X position in t', 'value': 'xt'},
           {'label': 'Y Position in t', 'value': 'yt'},
           {'label' : 'Velocity', 'value': 'vt'},
            {'label' : 'Acceleration', 'value': 'a'}
       ], value=['a']

    )
])


@app.callback(
    dash.dependencies.Output('graph-with-slider', 'figure'),
    [dash.dependencies.Input('Radio', 'value')])

def update_figure(selected_option):
    print(str(selected_option))
    #filtered_df = dM[selected_option]

    traces=[]
    traces.append(go.Scatter(
            x=dM.index,
            y=time_var[str(selected_option)],
            text=selected_option[0],
            mode='markers',
            opacity=0.7,
            marker={
                'size': 15,
                'line': {'width': 0.5, 'color': 'white'}
            },
            name=i
        ))


    return {
        'data': traces,
        'layout': go.Layout(
            xaxis={'title': 'Time'},
            yaxis={'title': selected_option},
            legend={'x': 0, 'y': 1},
            hovermode='closest'
        )
    }


if __name__ == '__main__':
    app.run_server()