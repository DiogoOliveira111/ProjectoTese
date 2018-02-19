import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
import easygui
import pickle
from tkinter import Tk, Label
from WBMTools.sandbox.interpolation import interpolate_data
traces =[]
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
vars={'time_var': time_var, 'space_var': space_var}
app = dash.Dash()

colors = {
    'background': '#ffffff',
    'text': '#7FDBFF'
}

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        children='Hello',
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
                    y = MouseDict['y'],
                    x= MouseDict['x'],
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

    )),


    html.Div([
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
       ], value='a'

    ),
    html.Button('Interpolate', id='interpolate', type='submit'),
    html.Div(id='output_spacevar',
             children='')

])])

@app.callback(
    dash.dependencies.Output('graph-with-slider', 'figure'),
    [dash.dependencies.Input('Radio', 'value')])
def update_figure(selected_option):

    #print(space_var)
    #print(len(time_var['xt']))
    #filtered_df = dM[selected_option]

    traces=[]
    traces.append(go.Scatter(
            x=time_var['tt'],
            y=time_var[str(selected_option)],
            text=selected_option[0],
            opacity=0.7,
            # marker={
            #     'size': 5,
            #     'line': {'width': 0.5, 'color': 'white'}
            # },
            line= {'width': 2, 'color': 'black'},
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

# @app.callback(
#     dash.dependencies.Output('PosGraf', 'figure'),
#     [dash.dependencies.Event('interpolate', 'n_clicks')])

@app.callback(dash.dependencies.Output('output_spacevar', 'children'),
              [dash.dependencies.Input('interpolate', 'n_clicks')])
def display_spacevar(n_clicks):
    return "length of strokes: ["
        # + min(space_var['l_strokes'])+ ","+ max(space_var['l_strokes'])+"\n"+\
         #  "Straightness: ["+ min(space_var['straightness']) + ","+ max(space_var['straightness'])+"\n"



if __name__ == '__main__':
    app.run_server()