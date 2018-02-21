from dash.dependencies import Input, Output, State, Event
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
import easygui
import pickle
from tkinter import Tk, Label
from WBMTools.sandbox.interpolation import interpolate_data
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
traces =[]
prevb1=0
prevb2=0
prevb3=0
finalStr=""
from scipy import signal
from scipy.signal import filtfilt

#Extra Functions

def lowpass(s, f, order=2, fs=1000.0, use_filtfilt=True):
    """
    @brief: for a given signal s rejects (attenuates) the frequencies higher
    then the cuttof frequency f and passes the frequencies lower than that
    value by applying a Butterworth digital filter
    @params:
    s: array-like
    signal
    f: int
    the cutoff frequency
    order: int
    Butterworth filter order
    fs: float
    sampling frequency
    @return:
    signal: array-like
    filtered signal
    """
    b, a = signal.butter(order, f / (fs / 2))

    if use_filtfilt:
        return filtfilt(b, a, s)

    return signal.lfilter(b, a, s)

def highpass(s, f, order=2, fs=1000.0, use_filtfilt=True):
    """
    @brief: for a given signal s rejects (attenuates) the frequencies lower
    then the cuttof frequency f and passes the frequencies higher than that
    value by applying a Butterworth digital filter
    @params:
    s: array-like
    signal
    f: int
    the cutoff frequency
    order: int
    Butterworth filter order
    fs: float
    sampling frequency
    @return:
    signal: array-like
    filtered signal
    """

    b, a = signal.butter(order, f * 2 / (fs / 2), btype='highpass')
    if use_filtfilt:
        return filtfilt(b, a, s)

    return signal.lfilter(b, a, s)

def bandpass(s, f1, f2, order=2, fs=1000.0, use_filtfilt=True):
    """
    @brief: for a given signal s passes the frequencies within a certain range
    (between f1 and f2) and rejects (attenuates) the frequencies outside that
    range by applying a Butterworth digital filter
    @params:
    s: array-like
    signal
    f1: int
    the lower cutoff frequency
    f2: int
    the upper cutoff frequency
    order: int
    Butterworth filter order
    fs: float
    sampling frequency
    @return:
    signal: array-like
    filtered signal
    """
    b, a = signal.butter(order, [f1 * 2 / fs, f2 * 2 / fs], btype='bandpass')

    if use_filtfilt:
        return filtfilt(b, a, s)

    return signal.lfilter(b, a, s)

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
html.Div(
    dcc.RadioItems(id='Radioheatmap',
       options=[
           {'label': 'Positional Map', 'value': 'xy'},
           {'label': 'Heatmap', 'value': 'heat'},
       ], value='xy'

    )),


    html.Div([
    dcc.Graph(id='timevar_graph'),
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
    html.Button('Interpolate', id='interpolate'),
    html.Div(id='output_spacevar'),
             dcc.Markdown(id='text_spacevar')

]),
    html.Div([
        html.Button('HighPass(H)', id='highpass', value=''),
        html.Button('LowPass(L)', id='lowpass'),
        html.Button('BandPass(Bp)', id='bandpass')]
),
    html.Div(dcc.Input(id='PreProcessing',
    placeholder='Enter a value...',
    type='text',
    value=''
)),
    html.Div([
        html.Button('Pre-Processing', id='preprocess'),
        dcc.Graph(id='timevar_graph_PP')
    ])

])

@app.callback(
    dash.dependencies.Output('timevar_graph_PP', 'figure'),
    [dash.dependencies.Input('preprocess', 'n_clicks'),
    dash.dependencies.Input('PreProcessing', 'value'),
    dash.dependencies.Input('timevar_graph', 'figure')]
)

def PreProcessStringParser(n_clicks, parse, data):
    if(n_clicks!=None):
        CutString=parse.split()
        for i in range(len(CutString)-1):
            if(CutString[i] =='H'):
                #function high pass
                print(data)
                data=highpass(data,int(CutString[i+1]))
                print('asasasas')
                # pass
            elif(CutString[i] =='L'):
                #Function Low Pass
                print(2)
                # pass
            elif (CutString[i] == 'BD'):
                #Function Band Pass
                print(3)
                # pass
    return data

@app.callback(
    dash.dependencies.Output('PreProcessing', 'value'),[
    dash.dependencies.Input('highpass', 'n_clicks'),
    dash.dependencies.Input('lowpass', 'n_clicks'),
    dash.dependencies.Input('bandpass', 'n_clicks')],
    [dash.dependencies.State('PreProcessing', 'value')]
    # prev_inputs=[
    #         dash.dependencies.PrevInput('button-1', 'n_clicks'),
    #         dash.dependencies.PrevInput('button-2', 'n_clicks'),
    #         dash.dependencies.PrevInput('button-3', 'n_clicks'),
    #         ]
)



def PreProcessingWrite(b1,b2,b3, finalStr):
    global  prevb1, prevb2, prevb3 #banhada com variaveis globais para funcionar, convem mudar
    if(b1!= None): # tem o problema de nao limpar, se calhar precisa de um botao para limpar
        if(b1>prevb1):
            finalStr+= 'H '
        prevb1 = b1

    if (b2 != None):
        if (b2 >prevb2):
            finalStr += 'L '
        prevb2 = b2

    if (b3 != None):
        if (b3 >prevb3):
            finalStr +="BD "
        prevb3 = b3

    return finalStr


@app.callback(
    dash.dependencies.Output('timevar_graph', 'figure'),
    [dash.dependencies.Input('Radio', 'value')])
def update_figure(selected_option):

    #print(len(time_var['jerk']))
    #print(len(time_var['xt']))
    #filtered_df = dM[selected_option]
    if(str(selected_option) in ("vt, vx, vy")):
        traces=[]
        traces.append(go.Scatter(
                x=time_var['ttv'],
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
    elif(str(selected_option) in ("xt, yt, a, jerk")):
        traces = []
        traces.append(go.Scatter(
            x=time_var['tt'],
            y=time_var[str(selected_option)],
            text=selected_option[0],
            opacity=0.7,
            # marker={
            #     'size': 5,
            #     'line': {'width': 0.5, 'color': 'white'}
            # },
            line={'width': 2, 'color': 'black'},
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

@app.callback(
    dash.dependencies.Output('PosGraf', 'figure'),
    [dash.dependencies.Input('interpolate', 'n_clicks'),
     dash.dependencies.Input('Radioheatmap', 'value')])
def interpolate_graf(n_clicks, value):
    if(value=='xy'):
        # if(n_clicks == None):
        #     pass
        if (n_clicks!=None): #1ª vez n_clicks== None
            traces = []
            traces.append(go.Scatter(
                x=space_var['xs'],
                y=space_var['ys'],
                #text=selected_option[0],
                opacity=0.7,
                # marker={
                #     'size': 5,
                #     'line': {'width': 0.5, 'color': 'white'}
                # },
                line={'width': 2, 'color': 'black'},
                name=i
            ))

            return {
                'data': traces,
                'layout': go.Layout(
                    xaxis={'title': 'X position Interpolated'},
                    yaxis={'title': 'Y position Interpolated'},
                    legend={'x': 0, 'y': 1},
                    hovermode='closest'
                )
            }
    elif(value =='heat'):
        x=space_var['xs']
        y=space_var['ys']
        colorscale = ['#7A4579', '#D56073', 'rgb(236,158,105)', (1, 1, 0.2), (0.98, 0.98, 0.98)]
        # traces=[

        trace1 = go.Scatter(
            x=x, y=y, mode='markers', name='points',
            marker=dict(color='rgb(102,0,0)', size=5, opacity=0.4)
        )
        trace2 = go.Histogram2dcontour(
            x=x, y=y, name='density', ncontours=20,
            colorscale='Hot', reversescale=True, showscale=False
        )

        data = [trace1, trace2]

        layout = go.Layout(
            showlegend=False,
            autosize=True,
            # width=600,
            # height=550,
            xaxis=dict(
                domain=[0, 0.85],
                showgrid=False,
                zeroline=False
            ),
            yaxis=dict(
                domain=[0, 0.85],
                showgrid=False,
                zeroline=False
            ),
            margin=dict(
                t=50
            ),
            hovermode='closest',
            bargap=0,
            xaxis2=dict(
                domain=[0.85, 1],
                showgrid=False,
                zeroline=False
            ),
            yaxis2=dict(
                domain=[0.85, 1],
                showgrid=False,
                zeroline=False
            )
        )
            # go.Histogram2d(x=x, y=y,
            #                 #colorscale='YIGnBu',
            #                 zmax=10,
            #                 nbinsx=50,
            #                 nbinsy=50,
            #                 zauto=False)]
        return {
            'data': data,
            'layout': go.Layout(
                xaxis=dict(ticks='', showgrid=False, zeroline=False, nticks=20),
                yaxis=dict(ticks='', showgrid=False, zeroline=False, nticks=20),
                autosize=True,
                hovermode='closest',
            )
            }


@app.callback(dash.dependencies.Output('text_spacevar', 'children'),
              [dash.dependencies.Input('interpolate', 'n_clicks')])
def display_spacevar(n_clicks):
    if(n_clicks!= None):
        return "Length of strokes: [{0}, {1}] px/items/n" \
                "Straightness: [{0}, {1}] px/px/n " \
               "Jitter: [{4}]  /n" \
                "Angles: [{5}, {6}] /n" \
               "Angular Velocity (w): [{7}, {8}] /n"\
                "Curvature: [{9}, {10}] /n".format(
                str(round(min(space_var['l_strokes']),2)),
                str(round(max(space_var['l_strokes']), 2)),
                str(round(min(space_var['straightness']), 2)),
                str(round(max(space_var['straightness']), 2)),
                str(round(space_var['jitter'],3)),
                str(round(min(space_var['angles']), 2)),
                str(round(max(space_var['angles']), 2)),
                str(round(min(space_var['w']), 2)),
                str(round(max(space_var['w']), 2)),
                str(round(min(space_var['curvatures']), 2)),
                str(round(max(space_var['curvatures']), 2))
                                                )


        #        +"Straightness: ["+ str(round(min(space_var['straightness']),2)) + ","+ str(round(max(space_var['straightness']),2))+"]/n    " \
        #         + "Jitter: "+ str(space_var['jitter']) \
        #          + "Angles: ["+str(round(min(space_var['angles']),2))+","+ str(round(max(space_var['angles']),2))+"]/n" \
        #         + "Angular Velocity (w): ["+str(round(min(space_var['w']),2))+","+str(round(max(space_var['w']),2))+"]/n"\
        #          + "Curvature: ["+str(round(min(space_var['curvatures']),2))+","+str(round(max(space_var['curvatures']),2))+"]/n"
        # .format(str(round(min(space_var['l_strokes']),2)),
        #         str(round(max(space_var['l_strokes']), 2)),
        #         str(round(min(space_var['straightness']), 2)),
        #         str(round(max(space_var['straightness']), 2)),
        #         str(space_var['jitter']),
        #         str(round(min(space_var['angles']), 2)),
        #         str(round(max(space_var['angles']), 2)),
        #         str(round(min(space_var['w']), 2)),
        #         str(round(max(space_var['w']), 2)),
        #         str(round(min(space_var['curvatures']), 2)),
        #         str(round(max(space_var['curvatures']), 2))
        #         )
        #




if __name__ == '__main__':
    app.run_server()

