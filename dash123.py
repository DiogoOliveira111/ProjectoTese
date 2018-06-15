import numpy
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
import regex as re
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import json
import pandas as pd
from ProcessingMethods import smooth, lowpass, highpass, bandpass
from SymbolicMethods import DiffC, Diff2C, RiseAmp, AmpC, absAmp, findDuplicates
from AuxiliaryMethods import _plot, detect_peaks,merge_chars
import base64
import io
import dash_table_experiments as dt
import datetime
from numpy import diff
from plotly import tools
import pylab as pl
from itertools import groupby, count
from fpdf import FPDF
import xlwt

traces =[]
preva1= 0
preva2= 0
preva3= 0
preva4= 0
prevb1= 0
prevb2= 0
prevb3= 0
prevb4= 0
prevb5=0

lastclick=0
lastclick1=0
lastclick2=0
finalStr=""

# path = easygui.fileopenbox()
#
# with open(path, 'rb') as handle:
#     collection= pickle.load(handle)
#
#
flag=0
MouseTime=[]
MouseX=[]
MouseY=[]
MouseDict={'x' : 'None', 'y' :'None'
}
# #
# for i in collection:
#
#     event=collection[i]
#     if( event['Type']=='Mouse'):
#         data=event['Data'].split(';')
#         if (i==0):
#             initial_time = float(data[-1]) #nao devia ser 0 em vez de -1?
#             MouseTime.append(initial_time/1000)
#         else:
#             MouseTime.append((float(data[-1]) - initial_time) / 1000)
#         MouseX.append(float(data[2]))
#         MouseY.append(float(data[3]))
#         flag=1 #Flag to determine if there is Mouse data in the collection
#
#     if(flag==0):
#         root= Tk()
#
#         # Make window 300x150 and place at position (50,50)
#         root.geometry("600x300+50+50")
#
#         # Create a label as a child of root window
#         my_text = Label(root, text='The Collection chosen has no Mouse Data')
#         my_text.pack()
#         root.mainloop()
#         exit()
#
# MouseDict = dict(t=MouseTime, x=MouseX, y=MouseY)
# dM = pd.DataFrame.from_dict(MouseDict)
# time_var,space_var=interpolate_data(dM,t_abandon=20)
# vars={'time_var': time_var, 'space_var': space_var}
# print(space_var['l_strokes'])
# print(space_var['straightness'])
# print(space_var['jitter'])
#

app = dash.Dash()
app.config['suppress_callback_exceptions']=True
# app.scripts.config.serve_locally = True #no idea what this does
# app.config.supress_callback_exceptions=True

image_filename = 'logo_libphys.png' # replace with your own image
encoded_image = base64.b64encode(open(image_filename, 'rb').read())


#-----------------------------
#Style Buttons
#-----------------------------
styleB = {'margin-top':'7px', 'margin-left':'5px', 'margin-right':'5px', 'margin-bottom':'7px'}
# margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
colors = {
    'background': '#FFFFFF',
    'text': '#304D6D'
}

#-----------------------------
#Auxiliary Functions
#-----------------------------

def DrawShapes(matchInitial, matchFinal, datax, datay, index):
    shapes=[]
    # print(len(matchInitial[index]))
    for i in range(len(matchInitial[index])):
        shape={
            'type': 'rect',
            # x-reference is assigned to the x-values
            'xref': 'x',
            # y-reference is assigned to the plot paper [0,1]
            'yref': 'y',
            'x0': datax[matchInitial[index][i]],
            'y0': min(datay),
            'x1': datax[matchFinal[index][i]-1], #estava a saltar fora, preciso de ver bem
            'y1': max(datay),
            'fillcolor': '#A7CCED',
            'opacity': 0.7,
            'line': {
                'width': 1,
                'color': 'rgb(55, 128, 191)',
                }}
        shapes.append(shape)
    return shapes


def UpdateTimeVarGraph(traces, selected_option,clicks):
    global time_var
    global space_var
    global pauseVector
    global MouseDict
    global MouseDictOriginal
    global clickIndex
    global cutIndex
    global clickVector
    global questionTime
    global questionlist
    # print(MouseTime)
    # straightness_replicated=[]
    # # for i in range(len(space_var['straightness'])):
    # #     straightness_replicated.append([space_var['straightness']][i] * cutIndex[i])
    # # print('ay')
    # print(straightness_replicated[0])
    # print(selected_option)
    if (str(selected_option) in {'vt', 'vx', 'vy', 'a', 'jerk'})    :


        traces.append(go.Scatter(
            x=time_var['ttv'],
            y=time_var[selected_option],
            text=selected_option,
            opacity=0.7,
            # marker={
            #     'size': 5,
            #     'line': {'width': 0.5, 'color': 'white'}
            # },
            line={'width': 2},
            name=str(selected_option)
        ))
        # if (clicks == 1):
        #     traces.append(go.Scatter(
        #         x=time_var['ttv'][clickIndex],
        #         y=time_var[selected_option][clickIndex],
        #         text=selected_option,
        #         mode='markers',
        #         opacity=0.7,
        #         marker={
        #             'size': 5,
        #             'line': {'width': 0.5, 'color': 'white'}
        #         },
        #         # line={'width': 2},
        #         name=str(selected_option)
        #     ))

    elif (str(selected_option) in {'xt', 'yt'}):
        traces.append(go.Scatter(
            x=time_var['tt'],
            y=time_var[str(selected_option)],
            text=selected_option,
            opacity=0.7,
            # marker={
            #     'size': 5,
            #     'line': {'width': 0.5, 'color': 'white'}
            # },
            line={'width': 2},
            name=str(selected_option)
        ))
        # if (clicks == 1):
        #     traces.append(go.Scatter(
        #         x=time_var['tt'][clickIndex],
        #         y=time_var[str(selected_option)][clickIndex],
        #         text=selected_option,
        #         opacity=0.7,
        #         marker={
        #             'size': 5,
        #             'line': {'width': 0.5, 'color': 'white'}
        #         },
        #         # line={'width': 2},
        #         name=str(selected_option)
        #     ))
    elif (str(selected_option) in {'xs', 'ys'}):
        traces.append(go.Scatter(
            x=space_var['ts'],
            y=space_var[str(selected_option)],
            text=selected_option,
            opacity=0.7,
            # marker={
            #     'size': 5,
            #     'line': {'width': 0.5, 'color': 'white'}
            # },
            line={'width': 2},
            name=str(selected_option)
        )) #n sei ha correspondencia entre os indexes do click e as timevar
        # if (clicks == 1):
        #     traces.append(go.Scatter(
        #         x=MouseDict['t'][clickIndex],
        #         y=pauseVector,
        #         text=selected_option,
        #         opacity=0.7,
        #         marker={
        #             'size': 5,
        #             'line': {'width': 0.5, 'color': 'white'}
        #         },
        #         # line={'width': 2},
        #         name=str(selected_option)
        #     ))
    elif(str(selected_option) =='angles'):
        traces.append(go.Scatter(
            x=space_var['ts'][:-1],
            y=space_var[str(selected_option)],
            text=selected_option,
            opacity=0.7,
            # marker={
            #     'size': 5,
            #     'line': {'width': 0.5, 'color': 'white'}
            # },
            line={'width': 2},
            name=str(selected_option)
        ))
        # if (clicks == 1):
        #     traces.append(go.Scatter(
        #         x=MouseDict['t'][clickIndex],
        #         y=pauseVector,
        #         text=selected_option,
        #         opacity=0.7,
        #         marker={
        #             'size': 5,
        #             'line': {'width': 0.5, 'color': 'white'}
        #         },
        #         # line={'width': 2},
        #         name=str(selected_option)
        #     ))
    elif(str(selected_option)=='curvatures'):
        traces.append(go.Scatter(
            x=space_var['ts'][:-2],
            y=space_var[str(selected_option)],
            text=selected_option,
            opacity=0.7,
            # marker={
            #     'size': 5,
            #     'line': {'width': 0.5, 'color': 'white'}
            # },
            line={'width': 2},
            name=str(selected_option)
        ))
    elif (str(selected_option) == 'var_curvatures'):
        traces.append(go.Scatter(
            x=space_var['ts'][:-3],
            y=space_var[str(selected_option)],
            text=selected_option,
            opacity=0.7,
            # marker={
            #     'size': 5,
            #     'line': {'width': 0.5, 'color': 'white'}
            # },
            line={'width': 2},
            name=str(selected_option)
        ))
        # if (clicks == 1):
        #     traces.append(go.Scatter(
        #         x=MouseDict['t'][clickIndex],
        #         y=pauseVector,
        #         text=selected_option,
        #         opacity=0.7,
        #         marker={
        #             'size': 5,
        #             'line': {'width': 0.5, 'color': 'white'}
        #         },
        #         # line={'width': 2},
        #         name=str(selected_option)
        #     ))
    elif (str(selected_option) == 'w'):
        traces.append(go.Scatter(
            x=space_var['ts'][:-2],
            y=space_var[str(selected_option)],
            text=selected_option,
            opacity=0.7,
            # marker={
            #     'size': 5,
            #     'line': {'width': 0.5, 'color': 'white'}
            # },
            line={'width': 2},
            name=str(selected_option)
        ))
    elif (str(selected_option) == 'pauses'):
        traces.append(go.Scatter(
            x=MouseDict['t'][1:],
            y=pauseVector,
            text=selected_option,
            opacity=0.7,
            marker={
                'size': 5,
                'line': {'width': 0.5, 'color': 'white'}
            },
            # line={'width': 2},
            name=str(selected_option)
        ))
        # if (clicks == 1):
        #     traces.append(go.Scatter(
        #         x=MouseDict['t'][:-1][clickIndex],
        #         y=pauseVector[clickIndex],
        #         text=selected_option,
        #         opacity=0.7,
        #         marker={
        #             'size': 5,
        #             'line': {'width': 0.5, 'color': 'white'}
        #         },
        #         # line={'width': 2},
        #         name=str(selected_option)
        #     ))
    elif (str(selected_option)=='straight'):


        size_time=len(MouseDict['t'])
        straightness=np.zeros(size_time)

        for i in range(len(cutIndex)):
            if i ==0:
                straightness[0:cutIndex[i]]=space_var['straightness'][0]

            elif i==len(cutIndex)-1:
                straightness[cutIndex[i]:size_time] = space_var['straightness'][i]

            else:
                straightness[cutIndex[i]:cutIndex[i+1]] = space_var['straightness'][i]


        traces.append(go.Scatter(
            x=MouseDict['t'],
            y=straightness,
            text=selected_option,
            opacity=0.7,
            marker={
                'size': 5,
                'line': {'width': 0.5, 'color': 'white'}
            },
            # line={'width': 2},
            name=str(selected_option)))

    elif (str(selected_option)=='lenStrokes'):

        size_time=len(MouseDict['t'])
        lenStrokes=np.zeros(size_time)
        for i in range(len(cutIndex)):
            if i ==0:
                lenStrokes[0:cutIndex[i]]=space_var['l_strokes'][0]

            elif i==len(cutIndex)-1:
                lenStrokes[cutIndex[i]:size_time] = space_var['l_strokes'][i]

            else:
                lenStrokes[cutIndex[i]:cutIndex[i+1]] = space_var['l_strokes'][i]

        traces.append(go.Scatter(
            x=MouseDict['t'],
            y=lenStrokes/1000,
            text=selected_option,
            opacity=0.7,
            marker={
                'size': 5,
                'line': {'width': 0.5, 'color': 'white'}
            },
            # line={'width': 2},
            name=str(selected_option)))

    elif (str(selected_option) == 'pausescumsum'):
        pauseVector = np.cumsum(pauseVector)
        traces.append(go.Scatter(
            x=MouseDict['t'][:-1],
            y=pauseVector,
            text=selected_option,
            opacity=0.7,
            marker={
                'size': 5,
                'line': {'width': 0.5, 'color': 'white'}
            },
            # line={'width': 2},
            name=str(selected_option)
        ))
    elif (str(selected_option) == 'time'):
        traces.append(go.Scatter(
            x=MouseDict['t'],
            y=MouseDict['t'],
            text=selected_option,
            opacity=0.7,
            marker={
                'size': 5,
                'line': {'width': 0.5, 'color': 'white'}
            },
            # line={'width': 2},
            name=str(selected_option)
        ))
    elif (str(selected_option) == 'clicks'):
        traces.append(go.Scatter(
            x=MouseDictOriginal['t'],
            y=clickVector,
            text=selected_option,
            opacity=0.7,
            marker={
                'size': 5,
                'line': {'width': 0.5, 'color': 'white'}
            },
            # line={'width': 2},
            name=str(selected_option)
        ))
    elif (str(selected_option) == 'ss'):
        traces.append(go.Scatter(
            x=space_var['ts'],
            y=space_var[str(selected_option)],
            text=selected_option,
            opacity=0.7,
            marker={
                'size': 5,
                'line': {'width': 0.5, 'color': 'white'}
            },
            # line={'width': 2},
            name=str(selected_option)
        ))
    elif (str(selected_option) == 'question'):
        traces.append(go.Scatter(
            x=questionTime,
            y=questionlist,
            text=selected_option,
            opacity=0.7,
            marker={
                'size': 5,
                'line': {'width': 0.5, 'color': 'white'}
            },
            # line={'width': 2},
            name=str(selected_option)
        ))
    elif (str(selected_option) == 'x'):
        traces.append(go.Scatter(
            x=MouseDict['t'],
            y=MouseDict['x'],
            text=selected_option,
            opacity=0.7,
            marker={
                'size': 5,
                'line': {'width': 0.5, 'color': 'white'}
            },
            # line={'width': 2},
            name=str(selected_option)
        ))
    elif (str(selected_option) == 'y'):
        traces.append(go.Scatter(
            x=MouseDict['t'],
            y=MouseDict['y'],
            text=selected_option,
            opacity=0.7,
            marker={
                'size': 5,
                'line': {'width': 0.5, 'color': 'white'}
            },
            # line={'width': 2},
            name=str(selected_option)
        ))

    return traces

def createLayoutTimevar(value, clicks, traces):
    global clickIndex
    global MouseDict
    global MouseDictOriginal
    Titles=dict(vt='Velocity in time',
         vy='Velocity in Y',
         vx='Velocity in X',
         jerk='Jerk',
         a='Acceleration in time',
         xt='X position in time',
         yt='Y position in time',
        xs='X interpolated in time',
        ys='Y interpolated in time',
        curvatures='Curvatures',
        var_curvatures='Variation of Curvature',
        w='Angular Velocity',
        angles='Angles',
        pauses='Time of pauses',
        straight= 'Straightness',
        lenStrokes='Length of Strokes',
        pausescumsum='Cumulative Sum of Pauses',
        time='Time passed',
        clicks='Clicks',
        question='Questions',
        ss='Space Traveled',
        x='X position in time not interpolated',
        y='y position in time not interpolated'
         )
    shapes = []
    if clicks==1:

        for i in clickIndex:
            shape={
                'type': 'line',
                'x0':MouseDictOriginal['t'][i],
                'y0': max(traces[0]['y']),
                'x1': MouseDictOriginal['t'][i],
                'y1': min(traces[0]['y']),
                'fillcolor': '#A7CCED',
                'opacity': 0.7,
                'line': {
                    'width': 1,
                    # 'color': 'rgb(55, 128, 191)',
                    }}
            shapes.append(shape)
    layout=go.Layout(
            shapes=shapes,
            # legend={'x': 0, 'y': 1},
            hovermode='closest',
            showlegend = True,
            autosize = True,
                            # width=600,
                            # height=550,
            title =Titles[str(value)] ,
            xaxis = dict(
            # domain=[0, 0.85],
                showgrid=False,
                zeroline=False,
                title='Time',
                titlefont=dict(
                family='Courier New, monospace',
                size=14,
                color='#7f7f7f'
                ),
            ),
            yaxis = dict(
            # domain=[0, 0.85],
                showgrid=False,
                zeroline=False,
                title=str(value),
                titlefont=dict(
                    family='Courier New, monospace',
                    size=14,
                    color='#7f7f7f'
                    ),
                ),
            margin = dict(
                    t=50
                ),
            bargap = 0,
        )
    return layout

#WRITE PDF FUNCTION

class PDF(FPDF):
    def header(self):
        # Logo
        self.image('libphyslogo.jpg', 10, 8, 33)
        # Arial bold 15
        self.set_font('Arial', 'B', 15)
        # Move to the right
        self.cell(80)
        # Title
        self.cell(30, 10, 'Report of Search Results', 1, 0, 'C')
        # Line break
        self.ln(20)

    # Page footer
    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Page number
        self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')


#------------------------------
# Plot vars - taboutput
#-------------------------------
Div_XY= dcc.Graph(
                id='PosGraph',
                figure={
                           'data': [
                               go.Scatter(
                                   y=MouseDict['y'],
                                   x=MouseDict['x'],
                                   mode='markers',
                                   name='Position',
                                   opacity=0.7,
                                   marker=dict(
                                       size=5,
                                       color='white',
                                       line=dict(
                                           width=1)

                                   )
                               )

                           ],
                           'layout': go.Layout(
                               showlegend=True,
                               xaxis={'title': 'X position'},
                               yaxis={'title': 'y position'},
                               margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                               legend={'x': 0, 'y': 1},
                               hovermode='closest')

                       },
                style={'display':'inline-block', 'width':'100%'})


Div_XY_checklist=dcc.Checklist(
    id='checklistheatmap',
    options=[
        {'label': 'Positional Map', 'value': 'XY'},
        {'label': 'Heatmap', 'value': 'heat'},
        {'label': 'Interpolate', 'value': 'interpolate'},
        {'label': 'Clicks', 'value': 'clicks'},
         {'label': 'Scroll', 'value': 'scroll'}
    ],
    values=['XY'],
    style={'display':'inline-block', 'width':'100%'})

Div_XY_uploadimage= dcc.Upload(html.Button('Upload Image'),
        id='upload-image',
        # children=html.Div([
        #     'Select Image File'
        # ]),
        style={
            'width': '50%',
            'height': '60px',
            # 'lineHeight': '60px',
            # 'borderWidth': '1px',
            # 'borderStyle': 'dashed',
            # 'borderRadius': '5px',
            'textAlign': 'center',

            'margin': '10px'
        },
    )

Div_Upload= dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px',
            'display':'inline-block'
        },
        # Allow multiple files to be uploaded
        multiple=True
    )

Div_XY_interpolate=dcc.Checklist(
    id='interpolatecheck',
    values=[],
    style={'display':'inline-block', 'width':'100%'}
)


Div_PP = dcc.Graph(id='timevar_graph_PP', style={'display': 'none', 'width':'100%'})

Div_SC = dcc.Graph(id='regexgraph', style={'display': 'none', 'width':'100%'}) # nao esta aqui a fazer nada

Div_S = html.Div([dcc.Graph(id='regexgraph', style={'display': 'none', 'width':'100%' }),
                    dcc.Dropdown( id='dropdown_Search',
                        options=[
                            # {'label':'Signal 1', 'value' :'S1'}
                            ],
                        value='',
                        # multi=True #tenho de implementar depois
                    )

                 ])

#-------------------------------------------------------------------------------
# App Layout
#-----------------------------------------------------------------------------


app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()),
             style={ 'width': '270px',
                     'height': '140px',
                     'align' :'left',
                    'margin-left':'5px',
                     'margin-right' : '20px',
                     'margin-top':'10px',
                      'margin-bottom': '15px',
                     'float': 'right'

                     # 'display': 'inline-block'
                     }
             ),

    html.H1(
        children='Symbolic Search in Web Monitoring Data',
        style={
            'textAlign': 'center',
            'color': colors['text'],
            'display': 'inline-block',
            'float':'left',
            'vertical-align':'text-middle'
        }
    ),

    # html.Div(children='Dash: A web application framework for Python.'),

    html.Div(
        children= [

            html.Div(
                [html.Div(children=[
                    Div_Upload,
                    dcc.Tabs(
                        tabs=[
                            {'label': 'Mouse Movement', 'value': 'XY'},
                            {'label': 'Pre-Processing', 'value': 'PP'},
                            {'label': 'Symbolic Connotation', 'value': 'SC'},
                            # {'label': 'Search', 'value': 'S'}
                        ],

                        value='XY',
                        id='tabs'
                    ),

                    html.Div(id='tab-output'
                             , children=[
                            html.Div([
                                Div_XY,
                                Div_XY_checklist,
                                Div_XY_uploadimage,


                                Div_XY_interpolate,
                                Div_PP,
                                Div_S], style={'width':'100%'})

                    ], style={'width':'99%', 'height' : '100%'}) # mudei o width de 100 para 99 e meti o height
                ])
            ], style={
                    'float':'left',
                    # 'display':'inline-block',
                       'width':'70%',
                        'backgroundColor': 'white',
                       })]),

            html.Div(children=[
                html.H4(
                        children='Pre-Processing Step',
                        style={
                            'textAlign': 'center',
                            'color': '#4C6684',
                            'display': 'inline-block',
                            'align':'center',
                            # 'vertical-align':'text-middle',
                            'margin-top':'7px',
                            'margin-left':'5px',
                            'margin-right':'5px',
                            'margin-bottom':'7px'

                        }
                    ),
                html.Div([
                    html.Button('H', id='highpass', value='', style=styleB, title='HighPass'),
                    html.Button('L', id='lowpass', style=styleB, title='LowPass'),
                    html.Button('BP', id='bandpass', style=styleB, title='BandPass'),
                    html.Button('S', id='smooth', style=styleB, title='Smooth'),
                    html.Button('ABS', id='absolute', style=styleB, title='Module'),]
                ),
                html.Div(id ='PreProcessDiv', children=[
                    dcc.Input(id='PreProcessing',
                               placeholder='Ex: "H 50 L 10"',
                               type='text',
                               value=''),
                    dcc.Input(id='PreProcessing1',
                               placeholder='Ex: "H 50 L 10"',
                               type='text',
                               value='',
                              style= {'display': 'none'}),
                    dcc.Input(id='PreProcessing2',
                               placeholder='Ex: "H 50 L 10"',
                               type='text',
                               value='',
                              style= {'display': 'none'}),

                    ]),
                html.Button('Pre-Processing', id='preprocess', style=styleB),

                html.H4(children='Symbolic Connotation Step',
                        style={
                            'textAlign': 'center',
                            'color': '#4C6684',
                            'display': 'inline-block',
                            'align':'left',
                            'vertical-align':'text-middle',
                            'margin-top':'7px',
                            'margin-left':'5px',
                            'margin-right':'5px',
                            'margin-bottom':'7px'

                        }
                    ),
                html.Div([
                    html.Button('%A', id='Amp', value='', style=styleB, title='Relative Amplitude'),
                    html.Button('↥A', id='relAmp', style=styleB, title='Absolute Amplitude'),
                    html.Button('1D', id='diff1', style=styleB, title='1st Derivative'),
                    html.Button('2D', id='diff2', style=styleB, title='2nd Derivative'),
                    html.Button('RA', id='riseamp', style=styleB, title='RiseAmp'),
                    html.Button('DUP', id='duplicate', style=styleB, title='Duplicate')
                ]
                ),
                html.Div(id='SymbolicConnotationDiv', children= [
                    dcc.Input(id='SCtext',
                            placeholder='Enter Symbolic Methods',
                            type='text',
                            value='',
                            # style={'display': 'none'}
                            ),
                    dcc.Input(id='SCtext1',
                            placeholder='Enter Symbolic Methods',
                            type='text',
                            value='',
                            style={'display': 'none'}
                            ),
                    dcc.Input(id='SCtext2',
                            placeholder='Enter Symbolic Methods',
                            type='text',
                            value='',
                            style={'display': 'none'}
                            )
                    ]
                         ),
                html.Div(
                    html.Button('Symbolic Connotation', id='SCbutton', style=styleB)),
                html.Div(id='SCresultDiv', children=[
                    dcc.Textarea(id="SCresult",
                                placeholder='Your symbolic series will appear here.',
                                value='',
                                # style={'margin-bottom':'7px'}
                                 ),
                    dcc.Textarea(id="SCresult1",
                                placeholder='Your symbolic series will appear here.',
                                value='',
                                style={'display':'none'}
                                 ),
                    dcc.Textarea(id="SCresult2",
                                placeholder='Your symbolic series will appear here.',
                                value='',
                                style={'display':'none'}
                                 ),
                    ]
                ),
                html.H4(children='Search Step',
                        style={
                            'textAlign': 'left',
                            'color': '#4C6684',
                            # 'display': 'inline-block',
                            'align':'left',
                            'vertical-align':'text-middle',
                            'margin-top':'7px',
                            'margin-left':'5px',
                            'margin-right':'5px',
                            'margin-bottom':'7px'

                        }
                    ),

                html.Div(id='regexSearchDiv', children=[
                        dcc.Input(id='regex',
                                    placeholder='Enter Regular Expression...',
                                    type='text',
                                    value='',
                            # style={'display': 'none'}
                        ),
                        dcc.Input(id='regex1',
                                    placeholder='Enter Regular Expression...',
                                    type='text',
                                    value='',
                                    style={'display': 'none'}
                                ),
                        dcc.Input(id='regex2',
                                  placeholder='Enter Regular Expression...',
                                  type='text',
                                  value='',
                                  style={'display': 'none'}
                              ),
                        dcc.Dropdown(
                            id='DropdownAndOr',
                            options=[
                                {'label': 'AND', 'value': 'AND'},
                                {'label': 'OR', 'value': 'OR'},
                            ],
                            value='OR',
                            searchable=False,
                        # style={'width': '100%'} # aparentemente nao da para mexer no estilo deste mambo
                        ),
                html.Button('Search Regex', id='searchregex', style=styleB),
                html.Button('Save Search in PDF', id='savePDF', style={'display': 'none'}),
                html.Button('Save Values in Excel', id='saveExcel', style={'display': 'none'})
                        ],
                         style={
                            'width': '25%',
                            'fontFamily': 'Sans-Serif',
                            'float' : 'left',
                            'backgroundColor': colors['background']},


        )], style={'float':'left', 'width':'20%', 'margin-left':'5%'}),

    html.Div(children=[
        dcc.Graph(id='timevar_graph', style={'width' : '100%'}),

        dcc.Dropdown( id='dropdown_timevar',
            options=[
                {'label': 'Velocity in X', 'value': 'vx'},
                {'label': 'Velocity in Y', 'value': 'vy'},
                {'label': 'Jerk', 'value': 'jerk'},
                {'label': 'X position in t', 'value': 'xt'},
                {'label': 'Y Position in t', 'value': 'yt'},
                {'label' : 'Velocity', 'value': 'vt'},
                {'label' : 'Acceleration', 'value': 'a'},
                {'label' : 'X interpolated in space', 'value': 'xs'},
                {'label' : 'Y interpolated in space', 'value': 'ys'},
                {'label' : 'Angles', 'value': 'angles'},
                {'label' : 'Curvatures', 'value': 'curvatures'},
                {'label' : 'Variation of Curvatures', 'value': 'var_curvatures'},
                {'label' : 'Angular Velocity', 'value': 'w'},
                {'label' : 'Pauses', 'value': 'pauses'},
                {'label' : 'Cumulative Sum of Pauses', 'value': 'pausescumsum'},
                {'label' : 'Straightness', 'value': 'straight'},
                {'label' : 'Length of Strokes', 'value': 'lenStrokes'},
                {'label' : 'Time Passed', 'value': 'time'},
                {'label' : 'Clicks', 'value': 'clicks'},
                {'label' : 'Question', 'value': 'question'},
                {'label' : 'Space Traveled', 'value': 'ss'},
                {'label' : 'X position in time not interpolated', 'value': 'x'},
                {'label' : 'Y position in time not interpolated', 'value': 'y'}
            ],
            multi=True,
            placeholder="",
            value=""
        )
    ], style={'display':'inline-block', 'width':'100%'}),
    html.Div(id='output-image-upload'),
    html.Div(dt.DataTable(rows=[{}]), style={'display': 'none'}),
    html.Div(id='hiddenDiv', style={'display':'none'}),
    html.Div(id='hiddenDiv_timevar', style={'display':'none'}),
    html.Div(id='hiddenDiv_Dictionary', style={'display':'none'}),
    html.Div(id='hiddenDiv_FinalString', style={'display': 'none'}),
    html.Div(id='hiddenDiv_PDF', style={'display': 'none'}),
    html.Div(id='hiddenDiv_PDFvalues', style={'display': 'none'}),
    html.Div(id='hiddenDiv_Excel', style={'display': 'none'}),
    html.Div([
        html.Button('Show Space Vars', id='showSpaceVar'),
        dcc.Markdown(id ='text_spacevar')]


    )
])


#TODO: POR ISTO NO SITIO
def longest_consecutive_increasing(intvect):

    max_so_far = 0
    curr_count = []
    max_pos = -1
    last_i=0
    #iterate through all integers in the vector
    for counter1 in range(len(intvect)):
        #case where we just started. No comparison needed yet.
        if counter1 == 0:
            pass
        #case where we have an increase

        elif (intvect[counter1]) == (last_i+1):
            curr_count.append(intvect[counter1]-2)
        last_i=intvect[counter1]

    return curr_count

def find_subsequences(lista):
    return [list(g) for k, g in groupby(lista, key=lambda n, c=count(): n - next(c))]

def obtainQuestion(df):

    Q=[]
    Q_int=[]
    TimeQ=[]
    Q_total_clicks=[]
    Q_total_int_answered=[]
    total_TQ_answered=[]
    Q_answered=[]
    TQ_answered=[]
    Q_int_answered=[]

    for index, i in df.iterrows():

        if ('java' in i[2]):
            q = re.search(r'Q\d{1,}', i[2]) #procura os valores numericos a seguir a Q
            # print(q.group())
            q = re.sub(r'0', '', q.group()[:-1]) + q.group()[-1:]
            if (int(q[1:]) > 18):
                continue
            else:
                # Q - string da questao (ex: Q1, Q2, Q3...)
                Q.append(q)
                # Q_int - questao em inteiro (ex: 1, 2, 3)
                Q_int.append(int(q[1:]))
                TimeQ.append(i[11])
                # quando houve click na questao
            if (i[1] == 1):
                Q_total_clicks.append(q)
                Q_total_int_answered.append(int(q[1:]))
                total_TQ_answered.append(i[11])
                # caso de questao mas hover na resposta (campo de respostas)
        elif ('answer' in i[2]):
            # procura questoes
            qa = re.search(r'Q\d{1,}', i[2])
            qa = re.sub(r'0', '', qa.group()[:-1]) + qa.group()[-1:]
            a = re.search(r'Q\d{1,}-(A\d)', i[2])
            # print(qa)
            # print(a)
            if (int(qa[1:]) > 18):
                continue
            else:
                Q.append(qa)
                Q_int.append(int(qa[1:]))
                TimeQ.append(i[11])
            if (i[1] == 1):
                # questao que foi respondida
                Q_answered.append(qa)
                Q_int_answered.append(int(qa[1:]))
                Q_total_clicks.append(qa)
                Q_total_int_answered.append(int(qa[1:]))

                TimeQ.append(i[11])
                TQ_answered.append(i[11])
                total_TQ_answered.append(i[11])

    return Q_int, TimeQ
#------------------------------------------------------
#   Callback Functions
#-----------------------------------------------------
def createDictionary(df):
    global time_var, space_var
    global cutIndex
    global MouseDict
    global clickIndex
    global pauseVector  # TODO fazer isto sem vars globais
    global Scroll
    global MouseDictOriginal
    global clickVector
    global questionTime
    global questionlist


    MouseX=[]
    MouseY = []
    MouseTime = []
    vars={}
    df.sort_values(by=[11])
    # print(diff(df[11]))
    # print(df.columns.tolist())
    # print(len(df[3]))
    MouseX=df[5]

    # x = [d for d in x if re.match('\d+', d)]  # procura na lista os que sao numeros, para retirar os undefined
    # MouseX = np.array(x).astype(int)  # converte string para int, so é preciso isto se o ficheiro tiver undefineds


    MouseY=df[6]
    MouseClicks=df[1]
    # print(MouseY)
    # MouseX=np.array(MouseX)
    # print(MouseX.iloc(1))
    # lastrowX=0
    # lastrowY=0
    # for rowX, rowY in zip(MouseX,MouseY):
    #     if rowX==lastrowX and rowY==lastrowY:
    #         MouseX.drop(rowX)
    #         MouseY.drop(rowY)
    #
    #     lastrowX=rowX
    #     lastrowY=rowY
    #
    #     print(rowX)
    #     print(rowY)
    #     # print(i)
        # if (MouseX.iloc == MouseX[i+1]):
        #     del MouseX.iloc[i+1]
        #     del MouseY[i+1]




    # y = [d for d in y if re.match('\d+', d)]  # procura na lista os que sao numeros, para retirar os undefined
    # MouseY= np.array(y).astype(int)

    for i in range(len(df[11])):
        if i==0:
            initial_time=df[11][0]/1000
            MouseTime.append(0)
        else:
            MouseTime.append((df[11][i]/1000)-initial_time)

    questionlist, questionTime = obtainQuestion(df) #para converter os timestamps das questoes
    for i in range(len(questionTime)):
        questionTime[i]=((questionTime[i]/1000)-initial_time)

    # MouseTime=MouseTime[::7]
    # print(questionTime)
    # print(MouseTime)
    MouseX = np.array(MouseX)
    MouseY = np.array(MouseY)
    MouseClicks=np.array(MouseClicks)
    # print(len(MouseClicks))
    # print(len(MouseTime))


#Encontrar os scroll events e o seu tamanho
    scrollIndex=[]
    scrollTime_partial=[]
    for i in range(len(MouseX)-1):
        if(MouseX[i]==MouseX[i+1]): # and MouseY[i]!=MouseY[i+1]: tirei isto porque ha pontos repetidos pelo meio dos scroll events
            scrollIndex.append(i)

    subsequences_scroll=find_subsequences(scrollIndex)

#Para acrescentar o ultimo ponto a cada subsequencia
    for i in range(len(subsequences_scroll)):
        for j in range(len(subsequences_scroll[i])):
            if j==len(subsequences_scroll[i])-1:
                subsequences_scroll[i].append(subsequences_scroll[i][j]+1)

#calcula o tempo de cada scroll
    scrollTime=[]
    partialScroll=0

    for i in range(len(subsequences_scroll)):
        for j in range(len(subsequences_scroll[i])):
            if subsequences_scroll[i][j]==subsequences_scroll[i][-1]:
                scrollTime.append(MouseTime[subsequences_scroll[i][j]]-MouseTime[subsequences_scroll[i][0]])

    # print(subsequences_scroll)
    # print('yaya')
    # print(scrollTime)
    scroll_flat = [item for sublist in subsequences_scroll for item in sublist]
    # print(scroll_flat)
#Encontrar os clicks, e saber a sua duraçao
    MouseClickDuration = []
    ClickCounter = -1 #last click is not considered, since it is necessary to leave the page
    for i in range(len(MouseClicks)-1):
        if MouseClicks[i] ==1 and MouseClicks[i+1]==4: #tem o problema de haver 1 seguido de 0 e depois 4
            MouseClickDuration.append(MouseTime[i+1]-MouseTime[i])
        if MouseClicks[i]==1:
            ClickCounter=ClickCounter+1
    # print(ClickCounter)
    # print(MouseClickDuration)
    # print(len(MouseClickDuration))

#criar o vector de clicks para as timevar
    clickVector=np.zeros(len(MouseTime))
    for i in range(len((MouseClicks))):
        if MouseClicks[i]==1 or MouseClicks[i]==4:
            clickVector[i]=1


#descobrir index de posiçoes duplicadas
    lista_index=[]
    for i in range(len(MouseX) - 1): #saber index de duplicates
        if (MouseX[i] == MouseX[i + 1]) and (MouseY[i] == MouseY[i + 1]):
           lista_index.append(i+1)

#eliminar as posiçoes com duplicates
    MouseXoriginal=MouseX
    MouseYoriginal = MouseY
    MouseTimeoriginal=MouseTime
    MouseX = np.delete(MouseX, lista_index) #remove duplicates
    MouseY = np.delete(MouseY, lista_index)
    MouseTime = np.delete(MouseTime, lista_index)
    # print(scroll_flat)

    # for i in lista_index:
    #     if i in scroll_flat:
    #         scroll_flat.remove(i)

    Scroll=scroll_flat
    # Scroll=np.delete(scroll_flat, lista_index)




    lista_incrementing=longest_consecutive_increasing(lista_index) #lista de duplicates seguidos TA MAL

#encontra as subsequencias na lista de duplicados
    subsequence_list=find_subsequences(lista_index)

#ha clicks no meio de posiçoes duplicadas, por isso é preciso ter em conta para nao os eliminar
    for i in subsequence_list:
        if len(i)>1: #ve as subsequences maiores que 1
            for j in range(len(i)):
                if(MouseClicks[i[j]] == 1): # ve se existe um click nessa subsequence
                    MouseClicks[i[0]-1]=1 # mete o click no index anterior ao primeiro(para n ser eliminado)
        elif len(i)==1:
            if MouseClicks[i]==1: #no caso de haver um click no index a ser eliminado
                MouseClicks[i[0]-1]=1




    # print(lista_incrementing)
    # for i in lista_incrementing:
    #     if MouseClicks[i]==1:
    #         print(i)
    #         print(MouseClicks[i])
    #         MouseClicks[i-1]=1

    clickIndex=[]
    ClickCounterOriginal=0
    for i in range(len(MouseClicks)):
        if MouseClicks[i]==1:
            clickIndex.append(i)
            ClickCounterOriginal =ClickCounterOriginal+1
    print('NR DE CLICKS')
    print(ClickCounterOriginal)

    MouseClicks=np.delete(MouseClicks, lista_index)


#saber o index dos clicks depois de removidos os duplicates
    # clickIndex=[]
    # for i in range(len(MouseClicks)):
    #     if MouseClicks[i]==1:
    #         clickIndex.append(i)
    # print(len(clickIndex))
    # print(clickIndex)
    # print('x new')
    # print(len(MouseX))
    # print('t new')
    # print(len(MouseTime))



    MouseTime.sort() #o sort do df nao esta a funcionar por alguma razao- isto garante que o MouseTime é sempre crescente

    numberStrokes=0
    cutTime=[]

#calcular o numero de strokes e as pausas entre cada um
    pauseVector = []
    for i in range(len(MouseTime)-1): #TODO: implement way for the user to control time that defines new stroke
        if MouseTime[i+1]-MouseTime[i]>1: #pauses>1 segundo => nova stroke
            numberStrokes=numberStrokes+1
            cutTime.append(MouseTime[i]) #append do ultimo ponto antes da pause
        pauseVector.append(MouseTime[i+1]-MouseTime[i]) #append dos tempos entre movimentos

    cutIndex=[]
    for i in range(len(cutTime)): #index das pauses
        cutIndex.append(np.argmax(MouseTime>cutTime[i]))

    lel=[]
    # for i in range(len(pauseVector)):
    #     if i==0:
    #         lel.append(np.where(MouseTime<= pauseVector[i]))
    #     elif i==len(pauseVector)-1:
    #         pass
    #     #     lel.append(np.where(MouseTime))
    #     else:
    #         lel.append(np.where(MouseTime<= pauseVector[i+1]-pauseVector[i]))
    # print('lel')
    # print(lel)

    pauseVectorIndex=[]
    pauseVectorcumSum=np.cumsum(pauseVector)
    # print(pauseVectorcumSum)
    # print(MouseTime)
    for i in range(len(pauseVectorcumSum)): #faz a procura dos indexes no vector MouseTime de cada pause, o comprimento de cada pause
        pauseVectorIndex.append(np.where(pauseVectorcumSum[i]>=MouseTime))


    # for i in range(len(pauseVectorIndex)):
    #     pauseVector_final=np.zeros(len(pauseVectorIndex[i]))
    #     if i == 0:
    #         max_index=max(pauseVectorIndex[0][i])
    #         pauseVector_final[0:max_index]=pauseVector[i]
    #     elif i==len(pauseVectorIndex)-1:
    #         max_index = max(pauseVectorIndex[0][i])
    #         pauseVector_final[max_index:-1]=pauseVector[i]
    #     else:
    #         print(i)
    #         max_index = max(pauseVectorIndex[0][i])
    #         max_index1 = max(pauseVectorIndex[0][i+1])
    #         pauseVector_final[max_index:max_index1] = pauseVector[i]
    #
    # print(pauseVector_final)
    # print(pauseVector_final)

    # print(numberStrokes)
    # print(cutTime)
    # print(cutIndex)
    # print(len(cutIndex))
    # print(len(MouseTime))
    # print(len(MouseTime))
    # print(len(pauseVector))


    MouseDictOriginal= dict(t=MouseTimeoriginal, x=MouseXoriginal, y=MouseYoriginal)
    MouseDict = dict(t=MouseTime, x=MouseX, y=MouseY)

    dM = pd.DataFrame.from_dict(MouseDict)
    time_var, space_var = interpolate_data(dM, t_abandon=20)

    # print(len(MouseTime))
    # print(len(time_var['ttv']))
    print(len(time_var['xt']))
    print(len(time_var['vt']))
    print(len(space_var['w']))
    print(len(space_var['ts']))
    print(space_var['ts'])

    print(len(space_var['straightness']))
    print(len(space_var['l_strokes']))
    # print(space_var['straightness'])
    # print(space_var['l_strokes'])
    # print((space_var['straightness'].tolist()).index(max(space_var['straightness'])))
    # print(np.cumsum(space_var['l_strokes'][0:12]))
    # a=np.cumsum(space_var['l_strokes'][0:12])
    # print(a[-1])
    # print(space_var['ss'].index(a[-1]))
    # print(len(MouseDict['x']))
    print(len(space_var['s']))
    print(len(space_var['ss']))
    print(len(space_var['xs']))
    print(len(space_var['angles']))
    print(len(space_var['curvatures']))
    print(len(space_var['var_curvatures']))
    print(len(space_var['w']))
    # print(len(time_var['tt']))
    # print(len(time_var['ttv']))
    # print(space_var['ss'])
    # print(space_var['s'])
    cleanedList = [x for x in space_var['straightness'] if str(x) != 'nan']

    # print(len(print(space_var['straightness'])))
    # print(len(time_var['vt']))

    vars = {'time_var': time_var, 'space_var': space_var}


def parse_contents(contents, filename, date):

    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)

    # try:
        # if 'csv' in filename:
            # Assume that the user uploaded a CSV file

    df = pd.read_csv(
        io.StringIO(decoded.decode('utf-8')), delimiter='\t', header=None,
        # names = ["N", "ID", "Type", "Xpos", "Ypos", 1, 2,3,4,5,6,7]
    )
    df.sort_values(by=[0])
# elif 'xls' in filename:
    createDictionary(df)
            # Assume that the user uploaded an excel file
            # df = pd.read_excel(io.BytesIO(decoded))


    # except Exception as e:
    #     print(e)
    #     return html.Div([
    #         'There was an error processing this file.'
    #     ])

    return html.Div([
            html.H5(filename),
            html.H6(datetime.datetime.fromtimestamp(date)),

            # Use the DataTable prototype component:
            # github.com/plotly/dash-table-experiments
            dt.DataTable(id='datatable', rows=df.to_dict('records')),

            html.Hr(),  # horizontal line

            # For debugging, display the raw contents provided by the web browser
            # html.Div('Raw Content'),
            # html.Pre(contents[0:200] + '...', style={
            #     'whiteSpace': 'pre-wrap',
            #     'wordBreak': 'break-all'
            # })
        ])


@app.callback(dash.dependencies.Output('output-image-upload', 'children'),
             [ dash.dependencies.Input('upload-data', 'contents'),
               dash.dependencies.Input('upload-data', 'filename'),
               dash.dependencies.Input('upload-data', 'last_modified')
              ])
def uploadData(list_of_contents, list_of_names, list_of_dates):
    children=[]

    if list_of_contents is not None:

        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]

    return children


@app.callback(
    dash.dependencies.Output('PosGraph', 'style'),
    [dash.dependencies.Input('tabs', 'value')]
)
def display_Pos(value):

    if value=='XY':
        return {'display': 'inline-block'}

    else:
        return {'display':'none'}

@app.callback(
    dash.dependencies.Output('checklistheatmap', 'style'),
    [dash.dependencies.Input('tabs', 'value')]
)
def display_checklist(value):

    if value=='XY':
        return {'display': 'inline-block'}

    else:
        return {'display':'none'}


@app.callback(
    dash.dependencies.Output('timevar_graph_PP', 'style'),
    [dash.dependencies.Input('tabs', 'value')])
def display_PP(value):
    if value=='PP':
        return {'display':'inline-block'}
    else:
        return {'display':'none'}

@app.callback(
    dash.dependencies.Output('regexgraph', 'style'),
    [dash.dependencies.Input('tabs', 'value')])
def display_S(value):
    if value=='S' or value=='SC':
        return {'display':'inline-block'}
    else:
        return {'display':'none'}

@app.callback(
    dash.dependencies.Output('savePDF', 'style'),
    [dash.dependencies.Input('hiddenDiv', 'children')]
)
def showSavebutton(matches):
    matches = json.loads(matches)

    if len(matches['matchInitial'] )!= 0 and len(matches['matchFinal'])!= 0:
        return {'display': 'inline-block'}
    else:
        return {'display':'none'}

@app.callback(
    dash.dependencies.Output('saveExcel', 'style'),
    [dash.dependencies.Input('hiddenDiv', 'children')])
def showSavebutton(matches):
    matches = json.loads(matches)

    if len(matches['matchInitial']) != 0 and len(matches['matchFinal']) != 0:
        return {'display': 'inline-block'}
    else:
        return {'display': 'none'}


@app.callback(
    dash.dependencies.Output('dropdown_Search', 'options'),
    [dash.dependencies.Input('hiddenDiv_timevar', 'children')],
    # [dash.dependencies.State('dropdown_Search', 'options')]
)
def updateDropdownSearch(selected_options):
    selected_options=json.loads(selected_options)
    print('dropdownsearch')
    print(selected_options)
    print(selected_options[0])
    timeVar_Dict=dict(vt='Velocity in time',
         vy='Velocity in Y',
         vx='Velocity in X',
         jerk='Jerk',
         a='Acceleration in time',
         xt='X position in time',
         yt='Y position in time',
         pauses='Pauses',
         straight='Straightness',
         lenStrokes='Length of Strokes',
        pausescumsum='Cumulative Sum of Pauses',
        time='Time Passed',
        clicks='Clicks',
        question='Questions',
        x='x in time not interpolated',
        y='y in time not interpolated',
      xs='X interpolated in space',
      ys='Y interpolated in space',
      angles='Angles',
      curvatures='Curvature',
      w='Angular Velocity',
      var_curvatures='Variation of Curvature',
      ss='Space Traveled'
         )
    # spaceVar_Dict=dict(
    #
    # )

    current_options=[]
    # print(selected_options)
    for i in range(len(selected_options)):
        if selected_options[i] in timeVar_Dict:

            current_options.append({'label': timeVar_Dict[str(selected_options[i])], 'value': selected_options[i]})
    if len(selected_options) == 2: #deve haver uma forma melhor de iterar isto para que independentemente da qtd de sinais escolhidos
            # for j in range(len(selected_options)):
        current_options.append({'label':timeVar_Dict[str(selected_options[0])]+ " and "+ timeVar_Dict[str(selected_options[1])], 'value': selected_options[0]+ " " +selected_options[1]})
    if len(selected_options)==3:
        current_options.append(
            {'label': timeVar_Dict[str(selected_options[0])] + " and " + timeVar_Dict[str(selected_options[1])],
             'value': selected_options[0] +" "+ selected_options[1]})
        current_options.append(
            {'label': timeVar_Dict[str(selected_options[1])] + " and " + timeVar_Dict[str(selected_options[2])],
             'value': selected_options[1] + " "+ selected_options[2]})
        current_options.append(
            {'label': timeVar_Dict[str(selected_options[0])] + " and " + timeVar_Dict[str(selected_options[2])],
             'value': selected_options[0] + " "+ selected_options[2]})
        current_options.append(
            {'label': timeVar_Dict[str(selected_options[0])] + " and " + timeVar_Dict[str(selected_options[1])]+ " and " + timeVar_Dict[str(selected_options[2])],
             'value': selected_options[0] + " " + selected_options[1]+ " "+ selected_options[2]})



    return current_options

@app.callback(
    dash.dependencies.Output('hiddenDiv_timevar', 'children'),
    [dash.dependencies.Input('dropdown_timevar', 'value')]
)
def updateTimeVar(value): #so uso isto 1x, talvez tenha de trocar
    return json.dumps(value)

@app.callback(
        dash.dependencies.Output('hiddenDiv', 'children'),
        [dash.dependencies.Input('regex', 'value'),
         dash.dependencies.Input('regex1', 'value'),
         dash.dependencies.Input('regex2', 'value'),
         dash.dependencies.Input('hiddenDiv_FinalString', 'value'),
         dash.dependencies.Input('searchregex', 'n_clicks'),
         ]
    )
def updateHiddenDiv(regex0, regex1, regex2,  string, n_clicks):
    global lastclick2
    matches={"matchInitial":[], "matchFinal":[]}
    regex=[]
    regex.extend((regex0, regex1, regex2))

    regex=list(filter(None, regex)) #para so iterar sobre as regex nao vazias

    if (n_clicks != None):
        if (n_clicks > lastclick2 and len(string)>0):
            matchInitial = [[],[],[]]
            matchFinal = [[],[],[]]
            # print(regex)
            # print(len(regex))
            # print(string)
            for j in range(len(regex)): #itera nos varios sinais
                regit = re.finditer(regex[j], string[j])

                for i in regit: #itera no proprio sinal
                    matchInitial[j].append((int(i.span()[0]))) #isto esta a dar a posiçao a começar em 1 por isso dava erro se houvesse match no ultimo ponto
                    matchFinal[j].append(int(i.span()[1]))

            matchInitial=list(filter(None, matchInitial))  #para remover as entries vazias
            matchFinal = list(filter(None, matchFinal))
            matchInitial = np.array(matchInitial)
            matchFinal = np.array(matchFinal)

            matches['matchInitial']=matchInitial.tolist()
            matches['matchFinal']= matchFinal.tolist()
            # print(len(string[0]))
            # print(matches)
    # print('hidden')
    print(matches)
    print(len(string[0]))
    return json.dumps(matches, sort_keys=True)

#Display hidden Content - PP, SC and regex

@app.callback(
    dash.dependencies.Output('regex1', 'style'),
    [dash.dependencies.Input('dropdown_timevar', 'value')
    ]
)
def showRegex1(selected_option):
    selected_option = np.array(selected_option)

    if selected_option.size > 1:

        return {'display': 'inline-block'}

    else:
        return {'display': 'none'}

@app.callback(
    dash.dependencies.Output('regex2', 'style'),
    [dash.dependencies.Input('dropdown_timevar', 'value')
    ]
)
def showRegex1(selected_option):
    selected_option = np.array(selected_option)

    if selected_option.size > 2:

        return {'display': 'inline-block'}

    else:
        return {'display': 'none'}



@app.callback(
    dash.dependencies.Output('SCresult1', 'style'),
    [dash.dependencies.Input('dropdown_timevar', 'value')]
)
def showtext1(selected_option):
    selected_option = np.array(selected_option)

    if selected_option.size>1:

        return {'display': 'inline-block'}

    else:
        return {'display':'none'}

@app.callback(
    dash.dependencies.Output('SCresult2', 'style'),
    [dash.dependencies.Input('dropdown_timevar', 'value')]
)
def showtext2(selected_option):
    selected_option = np.array(selected_option)
    if selected_option.size > 2:
        return {'display': 'inline-block'}

    else:
        return {'display': 'none'}


@app.callback(
    dash.dependencies.Output('regexgraph', 'figure'),
    [dash.dependencies.Input('hiddenDiv', 'children'),
     dash.dependencies.Input('searchregex', 'n_clicks'),
     dash.dependencies.Input('timevar_graph_PP', 'figure'),
     dash.dependencies.Input('dropdown_Search', 'value'),
    dash.dependencies.Input('hiddenDiv_timevar', 'children')
     ]
)
def RegexParser(matches, n_clicks, data, timevars_final, timevars_initial):
    global lastclick2
    global matches_final
    matches=json.loads(matches)
    # print('regexgraph')
    print(matches)

    timevars_initial=json.loads(timevars_initial)
    # print(timevars_initial)
    # print(timevars_final.split())
    timevars_final=timevars_final.split()

    if (n_clicks != None and len(timevars_final)>0):

        index_tv=[]
        for i in range(len(timevars_final)): #para percorrer os varios sinais e saber os indices originais
            index_tv.append(timevars_initial.index(timevars_final[i]))


        if(n_clicks>lastclick2 and len(matches)>0):

            traces_matches = []
            traces=[]



            counter_subplot = 1
            fig = tools.make_subplots(rows=len(index_tv), cols=1)
            for j in index_tv:


                matchInitial = np.array(matches['matchInitial'])
                matchFinal = np.array(matches['matchFinal'])
                print('merda funciona')
                print(matchInitial)
                print(matches)



                datax = np.array(data['data']['data'][j]['x'])
                datay= np.array(data['data']['data'][j]['y'])
                print('len')
                print(len(datax))
                matches_final=[]
                for i in range(len(matchFinal[j])): #cria uma lista com os indexes de todas as matches
                    matches_final.extend(range(matchInitial[j][i], matchFinal[j][i])) #nao considera a matchFinal como match, seria preciso por +1 aqui
                print('heckye')
                print(matches_final)
                # print(matches_final)
                if (len(matchInitial[j])>0 and len(matchFinal[j])>0 ):


                    traces_matches.append(go.Scatter(  # match initial append
                            x=datax[matches_final], #NOTA TENHO DE TER EM CONTA SE QUISER MOSTRAR UM SINAL NAO TRATADO
                            y=datay[matches_final],
                            # text=selected_option[0],
                            mode='markers',
                            opacity=0.7,

                            marker=dict(
                                # width=3,
                                # size=5,
                                color='#6666ff',
                                # line=dict(
                                #     width=2),


                            ),
                        connectgaps=False,
                        #     # xaxis='x'+ str(counter),
                        #     # yaxis='y1'+str(counter),
                            name='Match'

                        ))

                    # fig.append_trace(traces_matches[0], counter_subplot, 1)





                    # traces.append(go.Scatter( #match initial append
                    #     x=datax[matchInitial], #NOTA TENHO DE TER EM CONTA SE QUISER MOSTRAR UM SINAL NAO TRATADO
                    #     y=datay[matchInitial],
                    #     # text=selected_option[0],
                    #     mode='markers',
                    #     opacity=0.7,
                    #     marker=dict(
                    #         size=10,
                    #         color='#EAEBED',
                    #         line=dict(
                    #             width=2)
                    #
                    #     ),
                    #     # xaxis='x'+ str(counter),
                    #     # yaxis='y1'+str(counter),
                    #     name='Match Initial'
                    #
                    # ))
                    # # print('traces0')
                    # # print(traces[0])
                    # fig.append_trace(traces[0], counter_subplot, 1)
                    #
                    # traces.append(go.Scatter( #match final append
                    #     x=datax[matchFinal], #porque -1 ?
                    #     y=datay[matchFinal],
                    #     # text=selected_option[0],
                    #     mode='markers',
                    #     opacity=0.7,
                    #     marker=dict(
                    #         size=10,
                    #         color='#006989',
                    #         line=dict(
                    #             width=2)
                    #
                    #     ),
                    #     # xaxis='x' + str(counter),
                    #     # yaxis='y1' + str(counter),
                    #     name='Match Final'
                    # ))

                    # fig.append_trace(traces[1], counter_subplot, 1)
                traces.append(go.Scatter(
                    x=datax,
                    y=datay,
                    mode='lines',
                    line=dict(
                        color='black',
                        width=0.7

                    ),
                    # xaxis='x' + str(counter),
                    # yaxis='y1' + str(counter),
                    name='Signal'
                ))
                # print('counter')
                # print(counter_subplot)

                # print(traces[2])
                if(counter_subplot==1):
                    fig.append_trace(traces_matches[0], counter_subplot, 1) #subplot das matches
                    fig.append_trace(traces[0], counter_subplot, 1) #subplot do sinal
                elif(counter_subplot==2):
                    fig.append_trace(traces_matches[1], counter_subplot, 1)
                    fig.append_trace(traces[1], counter_subplot, 1)
                elif(counter_subplot==3):
                    fig.append_trace(traces_matches[2], counter_subplot, 1)
                    fig.append_trace(traces[2], counter_subplot, 1)


                counter_subplot = counter_subplot + 1
                # fig['layout'].update(height=600, width=600)



            return fig
            # {
            #         'data': traces,
            #         'layout': go.Layout(
            #             hovermode='closest',
            #             shapes =DrawShapes(matchInitial,matchFinal,datax, datay, j),
            #             autosize=True,
            #             xaxis=dict(ticks='', showgrid=False, zeroline=False),
            #             yaxis=dict(ticks='', showgrid=False, zeroline=False),
            #             yaxis2=dict(
            #                 domain=[0, 0.45],
            #                 anchor='x2'
            #             ),
            #             xaxis2=dict(anchor='y2')
            #         )}

        else:
            return {
                'data': data,
                'layout': go.Layout(
                    # legend={'x': 0, 'y': 1},
                    hovermode='closest',
                )
            }

    else:
        return {
            'data': data,
            'layout': go.Layout(
                # legend={'x': 0, 'y': 1},
                hovermode='closest',
            )
        }

@app.callback(
    dash.dependencies.Output('SCtext1', 'style'),
    [dash.dependencies.Input('dropdown_timevar', 'value')]
)
def showSC1(selected_option):
    selected_option = np.array(selected_option)
    if selected_option.size>1:
        return {'display': 'inline-block'}

    else:
        return {'display':'none'}

@app.callback(
    dash.dependencies.Output('SCtext2', 'style'),
    [dash.dependencies.Input('dropdown_timevar', 'value')]
)
def showSC1(selected_option):
    selected_option = np.array(selected_option)
    if selected_option.size>2:
        return {'display': 'inline-block'}

    else:
        return {'display':'none'}

#Symbolic Parsers

@app.callback(
    dash.dependencies.Output('SCresult2', 'value'),
    [dash.dependencies.Input('hiddenDiv_FinalString', 'value')]
)
def updateHiddenDivFinalStr(finalString):
    return finalString[2]

@app.callback(
    dash.dependencies.Output('SCresult1', 'value'),
    [dash.dependencies.Input('hiddenDiv_FinalString', 'value')]
)
def updateHiddenDivFinalStr(finalString):
    return finalString[1]

@app.callback(
    dash.dependencies.Output('SCresult', 'value'),
    [dash.dependencies.Input('hiddenDiv_FinalString', 'value')]
)
def updateHiddenDivFinalStr(finalString):
    return finalString[0]


def SCParser(parse, selector, data):
    finalString = [[],[],[]]
    # print(parse)
    # print(selector)
    # print(data)
    # finalString={'0':'',
    #              '1': '',
    #              '2':''}

    for j in range(selector):

        for i in range(len(parse[j])):

            if (parse[j][i] == '%A'):
                # function Amp
                finalString[j].append(AmpC(data['data']['data'][j]['y'], float(parse[j][i + 1])))
            elif (parse[j][i] == '↥A'):
                # function Amp
                finalString[j].append(absAmp(data['data']['data'][j]['y'], float(parse[j][i + 1]), float(parse[j][i + 2])))

            elif (parse[j][i] == '1D'):
                # Function 1st Derivative
                finalString[j].append(DiffC(data['data']['data'][j]['y'], float(parse[j][i + 1])))

            elif (parse[j][i] == '2D'):
                # Function 2nd Derivative
                finalString[j].append(Diff2C(data['data']['data'][j]['y'], float(parse[j][i + 1])))

            elif (parse[j][i] == 'R'):
                # Function RiseAmp
                finalString[j].append(RiseAmp(data['data']['data'][j]['y'], float(parse[j][i + 1])))  # nao sei se esta a fazer bem

            elif (parse[j][i] == 'DUP'):
                # Function 2nd Derivative
                print('ayyy')
                finalString[j].append(findDuplicates(data['data']['data'][j]['y']))
                print(finalString)

        finalString[j] = merge_chars(np.array(finalString[j]))
    print('pullup')
    print(finalString)
    return finalString

@app.callback(
    dash.dependencies.Output('SCtext', 'value'),[
    dash.dependencies.Input('Amp', 'n_clicks'),
    dash.dependencies.Input('diff1', 'n_clicks'),
    dash.dependencies.Input('diff2', 'n_clicks'),
    dash.dependencies.Input('riseamp', 'n_clicks'),
    dash.dependencies.Input('relAmp', 'n_clicks'),
    dash.dependencies.Input('duplicate', 'n_clicks')],
    [dash.dependencies.State('SCtext', 'value')]
)
def SymbolicConnotationWrite(a1,a2,a3,a4, a5,a6, finalStr):
    global  preva1, preva2, preva3, preva4, preva5, preva6 #banhada com variaveis globais para funcionar, convem mudar
    if(a1!= None): # tem o problema de nao limpar, se calhar precisa de um botao para limpar
        if(a1>preva1): #Amplitude
            finalStr+= '%A '
        preva1 = a1

    if (a2 != None): #1st derivative
        if (a2 >preva2):
            finalStr += '1D '
        preva2 = a2

    if (a3 != None): #2nd Derivative
        if (a3 >preva3):
            finalStr +="2D "
        preva3 = a3

    if (a4 != None): #RiseAmp
        if (a4 >preva4):
            finalStr +="R "
        preva4 = a4
    if (a5 != None):  # RiseAmp
        if (a5 > preva4):
            finalStr += "↥A "
        preva5 = a5
    if (a6 != None):  # Duplicates
        if (a6 > preva4):
            finalStr += "DUP "
        preva5 = a6
    return str(finalStr)


# @app.callback(
#     dash.dependencies.Output('SCresult2', 'value'),
#     [dash.dependencies.Input('SCresult', 'value')]
# )
# def UpdateSCresult1(finalString):
#
#     return finalString[2]
#
# @app.callback(
#     dash.dependencies.Output('SCresult1', 'value'),
#     [dash.dependencies.Input('SCresult', 'value')]
# )
# def UpdateSCresult1(finalString):
#
#     return finalString[1]


#Nota: input dos dados processados e nao originais
@app.callback(
    dash.dependencies.Output('hiddenDiv_FinalString', 'value'),
    [dash.dependencies.Input('SCbutton', 'n_clicks'),
    dash.dependencies.Input('timevar_graph_PP', 'figure')],
    [dash.dependencies.State('SCtext', 'value'),
     dash.dependencies.State('SCtext1', 'value'),
    dash.dependencies.State('SCtext2', 'value'),
    dash.dependencies.State('dropdown_timevar', 'value')
     ]
)
def SymbolicConnotationStringParser(n_clicks, data, parse, parse1, parse2, time_Vars):
    global lastclick1
    # finalString = {'0': '',
    #                '1': '',
    #                '2': ''}
    finalString = [[], [], []]


    if (n_clicks != None):

        # if(n_clicks>lastclick1): # para fazer com que o graf so se altere quando clicamos no botao e nao devido aos outros callbacks-grafico ou input box, BUG: ESTAVA A DEIXAR DE FUNCIONAR QUANDO CLICAVA NO "SCROLL" OU "CLICKS" PQ N ENTRAVA NO IF E FICAVA A STRING VAZIA

        CutString = []
        CutString.append(parse.split())
        CutString.append(parse1.split())
        CutString.append(parse2.split())
        selector=len(time_Vars)
        finalString=SCParser(CutString, selector, data)

            # for i in range(len(CutString)-1):
            #     if(CutString[i] =='A'):
            #         #function Amp
            #         finalString.append(AmpC(data['data']['data'][0]['y'],float(CutString[i+1])))
            #
            #     elif(CutString[i] =='1D'):
            #         #Function 1st Derivative
            #         finalString.append( DiffC(data['data']['data'][0]['y'], float(CutString[i + 1])))
            #
            #     elif (CutString[i] == '2D'):
            #         #Function 2nd Derivative
            #         finalString.append( Diff2C(data['data']['data'][0]['y'], float(CutString[i + 1])))
            #
            #     elif (CutString[i] == 'R'):
            #         #Function RiseAmp
            #         finalString.append(RiseAmp(data['data']['data'][0]['y'], float(CutString[i + 1]))) #nao sei se esta a fazer bem
          # finalString=merge_chars(np.array(finalString))  # esta a dar um erro estranho quando escrevo a caixa--FIXED
        lastclick1 = n_clicks
    print('len SC')
    print(len(data['data']['data'][0]['x']))
    # print(finalString)
    # print(len(finalString))
    # print(finalString)
    print(len(finalString[0]))
    return finalString


def PPProcessingParser(parse, selector, data, fs): #selector é para selecionar a variavel (time_var) a tratar

    for j in range(selector):

        for i in range(len(parse[j])):
            # print(data['data'][j]['y'])
            # print(parse[j])

            if (parse[j][i] == 'H'):
                # Function Amplitude

                data['data'][j]['y'] = highpass(data['data'][j]['y'], int(parse[j][i + 1]), fs=fs)  # aplicar o filtro aos dados y

            elif (parse[j][i] == 'L'):
                # Function Low Pass
                data['data'][j]['y'] = lowpass(data['data'][j]['y'], int(parse[j][i + 1]), fs=fs)

            elif (parse[j][i] == 'BP'):
                # Function Band Pass
                data['data'][j]['y'] = bandpass(data['data'][j]['y'], int(parse[j][i + 1]), int(parse[j][i + 2]), fs=fs)

            elif (parse[j][i] == 'S'):
                # Function Smooth
                smooth_signal = smooth(np.array(data['data'][j]['y']), int(parse[j][i + 1]))  # estranho porque o smooth retorna uma lista com os dados na 1ªpos e o valor do smooth na segunda
                data['data'][j]['y'] = smooth_signal

            elif (parse[j][i] == 'ABS'):
                # Function Absolute

                data['data'][j]['y'] = np.absolute(data['data'][j]['y'])



@app.callback(
    dash.dependencies.Output('timevar_graph_PP', 'figure'),
    [dash.dependencies.Input('preprocess', 'n_clicks'),
    dash.dependencies.Input('timevar_graph', 'figure')],
    [dash.dependencies.State('PreProcessing', 'value'),
     dash.dependencies.State('PreProcessing1', 'value'),
    dash.dependencies.State('PreProcessing2', 'value'),
    dash.dependencies.State('dropdown_timevar', 'value')
     ]
)
def PreProcessStringParser(n_clicks, data, parse, parse1, parse2, time_Vars):


    global lastclick
    # print(parse2)
    # print(data['data'][0])
    # print(data['data'][1])
    print('antespp')
    print(len(data['data'][0]['x']))
    if (n_clicks != None):
        if(n_clicks>lastclick): # para fazer com que o graf so se altere quando clicamos no botao e nao devido aos outros callbacks-grafico ou input box

            # maxTime=max(data['data'][0]['x'])
            # lengthOfSeries=len(data['data'][0]['x'])
            # fs=int(1/(maxTime/lengthOfSeries))

            fs=[]
            for i in range(len(data['data'][0]['x'])-1):
                fs.append(1/(data['data'][0]['x'][1]-data['data'][0]['x'][0]))
            fs=np.mean(fs) #para calcular a taxa de amostragem de forma um bocado manhosa
            CutString=[]
            CutString.append(parse.split())
            CutString.append(parse1.split())
            CutString.append( parse2.split())
            # print(CutString)
            # print(CutString[0])
            # print(CutString[1][1])
            selector=len(time_Vars)

            PPProcessingParser(CutString, selector, data, fs) #para simplificar o codigo

            # for i in range(len(CutString)): #estava range(len(CutString)-1 for some reason antes
            #     if(CutString[i] =='H'):
            #         #Function Amplitude
            #         data['data'][0]['y']=highpass(data['data'][0]['y'],int(CutString[i+1]), fs=fs) #aplicar o filtro aos dados y
            #
            #     elif(CutString[i] =='L'):
            #         #Function Low Pass
            #         data['data'][0]['y']=lowpass(data['data'][0]['y'], int(CutString[i+1]), fs=fs)
            #
            #     elif (CutString[i] == 'BP'):
            #         #Function Band Pass
            #         data['data'][0]['y']=bandpass(data['data'][0]['y'], int(CutString[i+1]), int(CutString[i+2]), fs=fs)
            #
            #     elif (CutString[i] == 'S'):
            #         #Function Smooth
            #         smooth_signal = smooth(np.array(data['data'][0]['y']), int(CutString[i+1])) #estranho porque o smooth retorna uma lista com os dados na 1ªpos e o valor do smooth na segunda
            #         data['data'][0]['y']=smooth_signal
            #
            #     elif (CutString[i] == 'ABS'):
            #         #Function Absolute
            #         data['data'][0]['y']=np.absolute(data['data'][0]['y'])
        print('len PP')
        print(len(data['data'][0]['x']))
        lastclick=n_clicks
    return {
        'data': data,
        'layout': go.Layout(
            xaxis=dict(ticks='', showgrid=False, zeroline=False, domain=[0, 1] ),
            yaxis=dict(ticks='', showgrid=False, zeroline=False, domain=[0, 1]),
            # autosize=True,
            hovermode='closest',
            )
        }

@app.callback(
    dash.dependencies.Output('PreProcessing', 'value'),[
    dash.dependencies.Input('highpass', 'n_clicks'),
    dash.dependencies.Input('lowpass', 'n_clicks'),
    dash.dependencies.Input('bandpass', 'n_clicks'),
    dash.dependencies.Input('smooth', 'n_clicks'),
    dash.dependencies.Input('absolute', 'n_clicks')],
    [dash.dependencies.State('PreProcessing', 'value')]
)
def PreProcessingWrite(b1,b2,b3,b4,b5, finalStr):
    global  prevb1, prevb2, prevb3, prevb4, prevb5 #banhada com variaveis globais para funcionar, convem mudar
    if(b1!= None): # tem o problema de nao limpar, se calhar precisa de um botao para limpar
        if(b1>prevb1): #o upgrade/downgrade fez com que isto ficasse a 1 cada vez que e clicado
            finalStr+= 'H '
        prevb1 = b1

    if (b2 != None):
        if (b2 >prevb2):
            finalStr += 'L '
        prevb2 = b2

    if (b3 != None):
        if (b3 >prevb3):
            finalStr += "BP "
        prevb3 = b3

    if (b4 != None):
        if (b4 >prevb4):
            finalStr +="S "
        prevb4 = b4

    if (b5 != None):
        if (b5 >prevb5):
            finalStr +="ABS "
        prevb5 = b5

    return finalStr

@app.callback(
    dash.dependencies.Output('PreProcessing2', 'style'),
    [dash.dependencies.Input('dropdown_timevar', 'value')]
)
def showPreProcess2(selected_option):
    selected_option = np.array(selected_option)
    if selected_option.size>2:
        return {'display': 'inline-block'}

    else:
        return {'display':'none'}


@app.callback(
    dash.dependencies.Output('PreProcessing1', 'style'),
    [dash.dependencies.Input('dropdown_timevar', 'value')]
)
def showPreProcess1(selected_option):
    selected_option = np.array(selected_option)

    if selected_option.size>1:
        return {'display': 'inline-block'}

    else:
        return {'display':'none'}

#para apagar as caixas qd se troca de timevar
@app.callback(
    dash.dependencies.Output('PreProcessing1', 'value'),
    [dash.dependencies.Input('dropdown_timevar', 'value')],
    [dash.dependencies.State('PreProcessing1', 'value')]
)
def cleanBoxPP1(selected_option, currentPP1):
    selected_option = np.array(selected_option)
    if selected_option.size<2:
        return ""
    else:
        return currentPP1

@app.callback(
    dash.dependencies.Output('PreProcessing2', 'value'),
    [dash.dependencies.Input('dropdown_timevar', 'value')],
    [dash.dependencies.State('PreProcessing2', 'value')]
)
def cleanBoxPP2(selected_option, currentPP2):
    selected_option = np.array(selected_option)
    if selected_option.size < 3:
        return ""
    else:
        return currentPP2

@app.callback(
    dash.dependencies.Output('SCtext1', 'value'),
    [dash.dependencies.Input('dropdown_timevar', 'value')],
    [dash.dependencies.State('SCtext1', 'value')]
)
def cleanBoxSC1(selected_option, currentSC1):
    selected_option = np.array(selected_option)
    if selected_option.size < 2:
        return ""
    else:
        return currentSC1

@app.callback(
    dash.dependencies.Output('SCtext2', 'value'),
    [dash.dependencies.Input('dropdown_timevar', 'value')],
    [dash.dependencies.State('SCtext2', 'value')]
)
def cleanBoxSC2(selected_option, currentSC2):
    selected_option = np.array(selected_option)
    if selected_option.size < 3:
        return ""
    else:
        return currentSC2

@app.callback(
    dash.dependencies.Output('regex1', 'value'),
    [dash.dependencies.Input('dropdown_timevar', 'value')],
    [dash.dependencies.State('regex1', 'value')]
)
def cleanBoxRegex1(selected_option, currentRegex1):
    selected_option = np.array(selected_option)
    if selected_option.size < 2:
        return ""
    else:
        return currentRegex1

@app.callback(
    dash.dependencies.Output('regex2', 'value'),
    [dash.dependencies.Input('dropdown_timevar', 'value')],
    [dash.dependencies.State('regex2', 'value')]
)
def cleanBoxRegex2(selected_option, currentRegex2):
    selected_option = np.array(selected_option)
    if selected_option.size < 3:
        return ""
    else:
        return currentRegex2

# @app.callback(
#     dash.dependencies.Output('PreProcessDiv', 'children'),
#     [dash.dependencies.Input('dropdown_timevar', 'value')]
# )
# def create_PP(selected_option):
#     Div=""
#
#     selected_option=np.array(selected_option)
#     print(len(selected_option))
#
#     for i in range(len(selected_option)):
#         if(len(selected_option)==1):
#             return html.Div( dcc.Input(id='PreProcessing',
#                           placeholder='LEL',
#                           type='text',
#                           value='')
#             )
#         elif(len(selected_option)==2):
#             print('hello')
#             return html.Div([dcc.Input(id='PreProcessing',
#                                       placeholder='LEL',
#                                       type='text',
#                                       value=''),
#                             html.Hr(),
#                             dcc.Input(id='PreProcessing1',
#                                       placeholder='LEL',
#                                       type='text',
#                                       value='')
#                             ])
#



@app.callback(
    dash.dependencies.Output('timevar_graph', 'figure'),
    [dash.dependencies.Input('dropdown_timevar', 'value'),
     dash.dependencies.Input('checklistheatmap', 'values')])
def update_timevarfigure(selected_option, values):


    traces=[]
    layout={}
    clicks=0
    if('clicks' in values):
        clicks=1

    if(len(selected_option) == 1):

        traces = UpdateTimeVarGraph(traces, selected_option[0], clicks)

        layout= createLayoutTimevar(selected_option[0], clicks, traces)

    if (len(selected_option)>1):
        for i in range(np.size(selected_option)):
            traces = UpdateTimeVarGraph(traces, selected_option[i], clicks)

        # for i in range(len(traces-1)):
        #     max_rel=0
        #     if max(traces[i]['y'])>len(traces[i+1]['y'])
        #         biggest_traces=i

        layout=createLayoutTimevar(selected_option[0], clicks, traces)

    return {
        'data':traces,
        'layout': layout
    }

@app.callback(
    dash.dependencies.Output('PosGraph', 'figure'),
    # [dash.dependencies.Input('interpolate', 'n_clicks'),
     [dash.dependencies.Input('checklistheatmap', 'values'),
     dash.dependencies.Input('hiddenDiv', 'children'),
     dash.dependencies.Input('hiddenDiv_timevar', 'children'),
    dash.dependencies.Input('DropdownAndOr', 'value'),
      dash.dependencies.Input('upload-image', 'contents')
     # dash.dependencies.Input('sliderpos', 'value')
     ])
def interpolate_graf(value, json_data, timevar, logic, image):
    global MouseDict
    global space_var
    global time_var
    global clickIndex
    global Scroll
    global MouseDictOriginal
    global questionTime

    # if image!=None:
    #     print('banana')
    matches = json.loads(json_data)


    timevar=json.loads(timevar)
    # print(timevar)
    # print(matches)
    matchInitial = matches['matchInitial']
    matchFinal = matches['matchFinal']

    traces=[]
    layout=[]
    for i in range(len(value)):
        if(value[i]=='XY'):
            traces.append(go.Scatter(
                y=MouseDict['y'],
                x=MouseDict['x'],
                name='Position',
                mode='markers',
                opacity=0.7,
                marker=dict(
                    size=5,
                    color='white',
                    line=dict(
                        width=1)
                ),


            ))
            # traces.append(go.Scatter(
            #     y=MouseDict['y'][427:435],
            #     x=MouseDict['x'][427:435],
            #     name='Position',
            #     mode='markers',
            #     opacity=0.7,
            #     marker=dict(
            #         size=5,
            #         color='red',
            #         line=dict(
            #             width=1)
            #             )
            #     ))
            layout= go.Layout(
                title='Positional Map',
                xaxis=dict(
                    title='X Axis',
                    titlefont=dict(
                        family='Courier New, monospace',
                        size=14,
                        color='#7f7f7f'
                    ),
                    ticks='', showgrid=False, zeroline=False),
                yaxis=dict(
                    title='Y Axis',
                    titlefont=dict(
                        family='Courier New, monospace',
                        size=14,
                        color='#7f7f7f'
                    ),
                    ticks='', showgrid=False, zeroline=False),
                autosize=True,
                hovermode='closest',
            )
            if image is not None: #N funciona qd tento fazer append de apenas a img ao layout
                layout=go.Layout(
                    title='Positional Map',
                    xaxis=dict(
                        title='X Axis',
                        titlefont=dict(
                            family='Courier New, monospace',
                            size=14,
                            color='#7f7f7f'
                        ),
                        ticks='', showgrid=False, zeroline=False),
                    yaxis=dict(
                        title='Y Axis',
                        titlefont=dict(
                            family='Courier New, monospace',
                            size=14,
                            color='#7f7f7f'
                        ),
                        ticks='', showgrid=False, zeroline=False),
                    autosize=True,
                    hovermode='closest',
                    images=[
                        dict(
                            source=image,
                            xref="x",
                            yref="y",
                            x=-2,
                            y=7,
                            sizex=10,
                            sizey=15,
                            sizing="stretch",
                            opacity=0.5,
                            layer="below"
                    )])

        if(value[i]=='interpolate'):
            traces.append(go.Scatter(
                x=time_var['xt'],
                y=time_var['yt'],
                # text=selected_option[0],
                opacity=0.7,
                # marker={
                #     'size': 5,
                #     'line': {'width': 0.5, 'color': 'white'}
                # },
                mode='lines',
                line={'width': 2, 'color': 'black'},
                name="Interpolated Position"
            ))
            layout = go.Layout(
                title='Interpolated Map',
                xaxis=dict(
                    title='X Axis',
                    titlefont=dict(
                        family='Courier New, monospace',
                        size=14,
                        color='#7f7f7f'
                    ),
                    ticks='', showgrid=False, zeroline=False),
                yaxis=dict(
                    title='Y Axis',
                    titlefont=dict(
                        family='Courier New, monospace',
                        size=14,
                        color='#7f7f7f'
                    ),
                    ticks='', showgrid=False, zeroline=False),
                autosize=True,
                hovermode='closest',
            )
            if image is not None:
                layout.append(go.Layout(
                    images=[dict(
                        source=image,
                        xref="x",
                        yref="y",
                        x=0,
                        y=3,
                        sizex=2,
                        sizey=2,
                        sizing="stretch",
                        opacity=0.5,
                        layer="below")]))

        if(value[i]=='heat'):
            x = space_var['xs']
            y = space_var['ys']
            colorscale = ['#7A4579', '#D56073', 'rgb(236,158,105)', (1, 1, 0.2), (0.98, 0.98, 0.98)]
            # traces=[

            trace = go.Histogram2dcontour(
                x=x,
                y=y,
                name='Position Density',
                ncontours=10,
                opacity=0.3,
                colorscale='YlGnBu',
                reversescale=True,
                showscale=True
            )

            traces = [trace]

            layout = go.Layout(
                showlegend=True,
                autosize=True,
                # width=600,
                # height=550,
                title='Heatmap',
                xaxis=dict(
                    # domain=[0, 0.85],
                    showgrid=False,
                    zeroline=False,
                    title='X axis',
                    titlefont=dict(
                        family='Courier New, monospace',
                        size=14,
                        color='#7f7f7f'
                    ),
                ),

                yaxis=dict(
                    # domain=[0, 0.85],
                    showgrid=False,
                    zeroline=False,
                    title='Y Axis',
                    titlefont=dict(
                        family='Courier New, monospace',
                        size=14,
                        color='#7f7f7f'
                    ),
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
            if image is not None:
                layout.append(go.Layout(
                    images=[dict(
                        source=image,
                        xref="x",
                        yref="y",
                        x=0,
                        y=3,
                        sizex=2,
                        sizey=2,
                        sizing="stretch",
                        opacity=0.5,
                        layer="below")]))
        if(value[i]=='test'):
            traces.append(go.Scatter(
                x=MouseDict['t'],
                y=MouseDict['y'],
                # text=selected_option[0],
                opacity=0.7,
                # marker={
                #     'size': 5,
                #     'line': {'width': 0.5, 'color': 'white'}
                # },
                line={'width': 2, 'color': 'black'},
                name="Interpolated Position"
            ))
        if (value[i] == 'clicks'):
            traces.append(go.Scatter(
                x=MouseDictOriginal['x'][clickIndex],
                y=MouseDictOriginal['y'][clickIndex],
                # text=selected_option[0],
                mode='markers',
                opacity=0.7,
                marker={
                    'symbol' : 'x',
                    'size': 10,
                    'line': {'width': 1, 'color': 'black'}
                },
                # line={'width': 2, 'color': 'black'},
                name="Clicks"
            ))
            # if image is not None:
            #     layout.append(go.Layout(
            #         images=[dict(
            #             source=image,
            #             xref="x",
            #             yref="y",
            #             x=0,
            #             y=3,
            #             sizex=2,
            #             sizey=2,
            #             sizing="stretch",
            #             opacity=0.5,
            #             layer="below")]))
        if (value[i] == 'scroll'):
            traces.append(go.Scatter(
                x=MouseDictOriginal['x'][Scroll],
                y=MouseDictOriginal['y'][Scroll],
                # text=selected_option[0],
                mode='markers',
                opacity=0.7,
                marker={
                    'symbol' : '.',
                    'size': 10,
                    'line': {'width': 1, 'color': 'black'}
                },
                # line={'width': 2, 'color': 'black'},
                name="Scroll"
            ))




    if(np.size(matches['matchInitial'])>0):
        print('aymatch')
        print(matches)
        lista_matches=[]
        # for i in range(len(matchInitial)): #No idea what the purpose of this is
        #     for a in range(matchInitial[i], matchFinal[i]+1): #+1 porque o range faz ate o valor-1
        #         lista_matches.append(a)

        nova_lista=[[],[],[]]
        flat_list= [[], [], []]

        # for i in range(len(lista_matches)):
        #     valor_time=time_var['ttv'][lista_matches[i]]
        #     nova_lista.append(MouseDict['t'].index(time_var['ttv'][lista_matches[i]]))

        time_array=np.array(MouseDict['t'])
        listA=["vt", "vx", "vy", "a", "jerk"]
        listB=["xt", "yt"]
        listC=['straight', 'lenStrokes', 'pausescumsum', 'time', 'clicks', 'x', 'y']
        listD=['curvatures', 'angles', 'w', 'var_curvatures', 'xs', 'ys', 'ss']

        # print(time_var['ttv'][matchInitial[0][0]])
        # print(time_var['ttv'][matchFinal[0][0]])
        # print(timevar)
        # print('matchup')
        # print(matches)
        # print(matchFinal)
        # print(matchInitial)
        # print(len(space_var['ts']))
        for j in range(len(timevar)): #para iterar entre os varios sinais


            if timevar[j] in listA:

                for i in range (len(matchInitial[j])): #para iterar dentro do mm sinal entre as varias matches
                    if i ==len(matchInitial[j])-1:
                        nova_lista[j].append(np.where( (time_array>= time_var['ttv'][matchInitial[j][i]]) & (time_array<= time_var['ttv'][matchFinal[j][i]-1]) ))
                    else:
                        nova_lista[j].append(np.where((time_array >= time_var['ttv'][matchInitial[j][i]]) & (time_array <= time_var['ttv'][matchFinal[j][i]])))

            if(timevar[j] in listB):
                for i in range (len(matchInitial[j])): #esta a sair fora do sinal qd a match é o ultimo ponto, acho que nao esta correcto assim
                    if i == len(matchInitial[j])-1:
                        nova_lista[j].append(np.where((time_array >= time_var['tt'][matchInitial[j][i]]) & (time_array <= time_var['tt'][matchFinal[j][i]-1])))#    TODO: REVER ESTES LIMITES FINAIS DAS MATCHES #é bem capaz de isto estar mal
                    else:
                        nova_lista[j].append(np.where( (time_array>= time_var['tt'][matchInitial[j][i]]) & (time_array<= time_var['tt'][matchFinal[j][i]]) )) #pus -1 porque estava a sair fora do vector

            if(timevar[j] in listC): #as matches ja sao dadas no vector de tempo mouseDict['t'], logo nao preciso de procurar
                for i in range(len(matchInitial[j])):
                    flat_list[j].extend(range(matchInitial[j][i], matchFinal[j][i]))

            if (timevar[j] in listD):
                print('len')
                print(len(matchInitial[j]))
                for i in range(len(matchInitial[j])):  # esta a sair fora do sinal qd a match é o ultimo ponto, acho que nao esta correcto assim
                    if i == len(matchInitial[j]) - 1:
                        nova_lista[j].append(np.where((time_array >= space_var['ts'][matchInitial[j][i]]) & (time_array <= space_var['ts'][matchFinal[j][i] - 1])))
                    else:
                        nova_lista[j].append(np.where((time_array >= space_var['ts'][matchInitial[j][i]]) & (time_array <= space_var['ts'][matchFinal[j][i]])))  # pus -1 porque estava a sair fora do vector

            if(timevar[j] == 'question'): #esta series tem um x diferente, que nao o MouseDict['t']
                for i in range (len(matchInitial[j])): #esta a sair fora do sinal qd a match é o ultimo ponto, acho que nao esta correcto assim
                    if i == len(matchInitial[j])-1:
                        nova_lista[j].append(np.where((time_array >= questionTime[matchInitial[j][i]]) & (time_array <= questionTime[matchFinal[j][i]-1])))
                    else:
                        nova_lista[j].append(np.where( (time_array>= questionTime[matchInitial[j][i]]) & (time_array<= questionTime[matchFinal[j][i]]) )) #pus -1 porque estava a sair fora do vector

            print('noval')
            print(nova_lista)
            if (len(nova_lista[j])!=0 ): #para fazer flat das matches, acho que se usar extend em vez de append isto fica desnecessario
                flat_list[j]=np.concatenate(nova_lista[j], axis=1)[0]
            print(flat_list)

        flat_list = np.array(flat_list)
        final_list = []

        for i in range(len(flat_list)): #para remover a parte vazia dos vectores pq o filter nao funciona
            if len(flat_list[i])>0:
                final_list.append(flat_list[i])
        print('old')
        print(final_list)





        #concatenate com ciclos
        # for i in range(len(nova_lista)):
        #     for a in range(len(nova_lista[i])):
        #         for b in range(len(nova_lista[i][a])):
        #             flat_list.append(nova_lista[i][a][b])
        # flat_list=  np.concatenate(nova_lista, axis=0)


        #aproximaçao do valor de tempo do outro vector
        # ratio=len(MouseDict['t'])/len(time_var['ttv'])
        # nova_lista = [i * ratio for i in lista_matches]
        # nova_lista=np.array((nova_lista))
        # nova_lista=np.rint(nova_lista).astype(int)
        # print(timevar)
        if 'clicks' not in timevar:  #o clicks tem de ser feito com o vector original,NOTE: ISTO PODE ESTAR MAL
            Xpos = np.array(MouseDict['x'])
            Ypos = np.array(MouseDict['y'])
        else:

            Xpos = np.array(MouseDictOriginal['x'])
            Ypos = np.array(MouseDictOriginal['y'])

        if (logic=='AND'): #HA UM ALGORITMO MELHOR PARA ISTO FOR SURE- encontrar os duplicates nas nested lists
            duplicates_list=[]
            auxiliary_list=[]
            for signal in range(len(final_list)):
                for i in range(len(final_list[signal])):
                    if final_list[signal][i] not in auxiliary_list:
                        auxiliary_list.append(final_list[signal][i])
                    elif final_list[signal][i] in auxiliary_list:
                        duplicates_list.append(final_list[signal][i])

            final_list=np.array(duplicates_list)
            # print('final')
            # print(final_list)
            traces.append(go.Scatter(
                x=Xpos[final_list],
                y=Ypos[final_list],
                # text=selected_option[0],
                opacity=1,
                mode='markers',
                # marker={
                #     'size': 5,
                #     'line': {'width': 0.5, 'color': 'white'}
                # },
                marker={'size': 5,
                        # 'color': '#A7CCED'
                        },
                name="Matches"
            ))





        elif(logic=='OR'):
            for i in range(len(final_list)):
                traces.append(go.Scatter(
                        x=Xpos[final_list[i]],
                        y=Ypos[final_list[i]],
                        # text=selected_option[0],
                        opacity=1,
                        mode='markers',
                        # marker={
                        #     'size': 5,
                        #     'line': {'width': 0.5, 'color': 'white'}
                        # },
                        marker={'size': 5,
                                # 'color': '#A7CCED'
                                },
                        name="Matches"+ str(i)
                    ))

    return {
            'data': traces,
            'layout': layout
    }

@app.callback(
    dash.dependencies.Output('hiddenDiv_Excel', 'value'),
    [
     dash.dependencies.Input('hiddenDiv_PDFvalues', 'children'),
        dash.dependencies.Input('saveExcel', 'n_clicks')]
)
def createExcel( info, clicks):
    global MouseDict
    global MouseDictOriginal
    dictSave = json.loads(info)
    clickValue = 0
    sheet='New sheet'
    timeVar_Dict = dict(vt='Velocity in time',
                        vy='Velocity in Y',
                        vx='Velocity in X',
                        jerk='Jerk',
                        a='Acceleration in time',
                        xt='X position in time',
                        yt='Y position in time',
                        pauses='Pauses',
                        straight='Straightness',
                        lenStrokes='Length of Strokes',
                        pausescumsum='Cumulative Sum of Pauses',
                        time='Time Passed',
                        clicks='Clicks',
                        xs='X interpolated in space',
                        ys='Y interpolated in space',
                        angles='Angles',
                        curvatures='Curvature',
                        w='Angular Velocity',
                        var_curvatures='Variation of Curvature'
                        )
    # final_listPOS = np.asarray(dictSave['final_listPOS'], dtype=np.int64)
    if clicks != None:
        book = xlwt.Workbook()
        sh = book.add_sheet(sheet)

        # variables = [x, y, z]
        signalString= 'Unprocessed Signal'
        signalPPString = 'Processed Signal'
        signalSCString = 'Symbolic Signal'
        signalMatchPos= 'Position of Matches'
        stringMatchX='X coordinate of match'
        stringMatchY = 'Y coordinate of match'
        # desc = [x_desc, y_desc, z_desc]

        # for n, v_desc, v in enumerate(zip(desc, variables)):
        #     sh.write(n, 0, v_desc)
        #     sh.write(n, 1, v)
        for i in range(len(dictSave['timevar'])):
            print('types')
            # print(MouseDict['x'][dictSave['final_listPOS'][i]].dtype)
            # print(MouseDict['x'][dictSave['final_listPOS']].dtype)
            # print(type(dictSave['final_listPOS'][i]))
            # print(type(dictSave['final_listPOS']))
            # M = np.asarray(dictSave['final_listPOS'][i], dtype=np.float64)
            MouseDict['x'].tolist()
            MouseDict['y'].tolist()
            MouseDictOriginal['x'].tolist()
            MouseDictOriginal['y'].tolist()
            print(i)
            print(len(dictSave['timevar']))
            print(dictSave['timevar'])
            # print(dictSave['Signals'])
            sh.write(0,0 + 9 *i, str(timeVar_Dict[dictSave['timevar'][i]]))
            sh.write(0, 1+ 9 *i, "Time")
            sh.write(0, 2+ 9 *i, signalString)
            sh.write(0, 3+ 9 *i, signalPPString)
            sh.write(0, 4+ 9 *i, signalSCString)
            sh.write(0, 5+ 9 *i, signalMatchPos)
            sh.write(0, 6+ 9 *i, stringMatchX)
            sh.write(0, 7+ 9 *i, stringMatchY)

            for m1, e1 in enumerate(dictSave['SignalsX'][i]):
                sh.write(m1+1, 1+ 9 *i, e1)
            for m2, e2 in enumerate(dictSave['SignalsY'][i]):
                sh.write(m2+1, 2+ 9 *i, e2)
            for m3, e3 in enumerate(dictSave['SignalsPP'][i]):
                sh.write(m3+1, 3+ 9 *i, e3)
            for m4, e4 in enumerate(dictSave['SC'][i]):
                sh.write(m4+1, 4+ 9 *i, e4)
            for m5, e5 in enumerate(dictSave['matches_final'][i]):
                sh.write(m5+1, 5+ 9 *i, e5)
            if dictSave['timevar'][i] == 'clicks':
                listPosXOriginal=MouseDictOriginal['x'][dictSave['final_listPOS'][i]]
                listPosYOriginal=MouseDictOriginal['y'][dictSave['final_listPOS'][i]]
                listPosXOriginal = listPosXOriginal.tolist()
                listPosYOriginal=listPosYOriginal.tolist()
                for m6, e6 in enumerate(listPosXOriginal):
                    sh.write(m6+1, 6+ 9 *i, e6)
                for m7, e7 in enumerate(listPosYOriginal):
                    sh.write(m7+1, 7+ 9 *i, e7)
            else:
                listPosX=MouseDict['x'][dictSave['final_listPOS'][i]].astype(numpy.float64) #nao aceita numpy array, tem de ser list
                listPosY=MouseDict['y'][dictSave['final_listPOS'][i]].astype(numpy.float64)
                listPosX=listPosX.tolist()
                listPosY=listPosY.tolist()
                for m6, e6 in enumerate(listPosX):
                    sh.write(m6+1, 6+ 9 *i, e6)
                for m7, e7 in enumerate(listPosY):
                    sh.write(m7+1, 7+ 9 *i, e7)

        book.save('FileExcel')
    clickValue = clicks
    return clicks

@app.callback(
    dash.dependencies.Output('hiddenDiv_PDF', 'value'),
    [
     dash.dependencies.Input('hiddenDiv_PDFvalues', 'children'),
        dash.dependencies.Input('savePDF', 'n_clicks')]
)
def createPDF( info, clicks):
    global clickValue
    global MouseDict
    global MouseDictOriginal

    timeVar_Dict = dict(vt='Velocity in time',
                        vy='Velocity in Y',
                        vx='Velocity in X',
                        jerk='Jerk',
                        a='Acceleration in time',
                        xt='X position in time',
                        yt='Y position in time',
                        pauses='Pauses',
                        straight='Straightness',
                        lenStrokes='Length of Strokes',
                        pausescumsum='Cumulative Sum of Pauses',
                        time='Time Passed',
                        clicks='Clicks',
                        xs='X interpolated in space',
                        ys='Y interpolated in space',
                        angles='Angles',
                        curvatures='Curvature',
                        w='Angular Velocity',
                        var_curvatures= 'Variation of Curvature'
    )
    dictSave=json.loads(info)
    # print(dictSave['timevar'][0])
    # print(dictSave)

    # print(len(dictSave['SCtext']))
    clickValue=0 #remove
    # print(clicks)
    # print(dictSave['final_listPOS'])
    if clicks!=None:

        # print(dictSave['Signals'])
        # Write PDF
        pdf = PDF()
        pdf.alias_nb_pages()
        pdf.add_page()
        pdf.set_font('Times', '', 12)
        for i in range(len(dictSave['SCtext'])): #para percorrer os varios sinais, podia ter escolhido outra var doesnt matter
            pdf.cell(0, 10, 'Signal Analysed: ' + str(timeVar_Dict[dictSave['timevar'][i]]), 0, 1)
            pdf.cell(0, 10, 'PreProcessing Method Used: ' + str(dictSave['PPtext'][i]), 0, 1) #TODO: pode estar vazio e depois da erro
            pdf.cell(0, 10, 'Symbolic Connotation Method Used: ' + str(dictSave['SCtext'][i]), 0, 1)
            pdf.cell(0, 10, 'Regex Searched: ' + str(dictSave['regex'][i]), 0, 1)
            pdf.cell(0, 10, 'Time: ' + str(dictSave['SignalsX'][i]), 0, 1)
            pdf.cell(0, 10, 'Unprocessed Signal: ' + str(dictSave['SignalsY'][i]), 0, 1)
            pdf.cell(0, 10, 'Processed Signal: ' + str(dictSave['SignalsPP'][i]), 0, 1)
            pdf.cell(0, 10, 'Number of Matches: ' + str(dictSave['NrofMatches'][i]), 0, 1)
            pdf.cell(0, 10, 'List of Matches Position: ' + str(dictSave['matches_final'][i]), 0, 1)
            pdf.cell(0, 10, '% of matches: ' + str(dictSave['matchPercentage'][i]), 0, 1)
            if dictSave['timevar'][i]=='clicks': #por causa da artimanha do MouseDictOriginal
                    pdf.cell(0, 10, 'Position of Matches: ', 0, 1)
                    pdf.cell(0, 10,'X: ' + str(MouseDictOriginal['x'][dictSave['final_listPOS'][i]]), 0, 1)
                    pdf.cell(0, 10, 'Y: ' + str(MouseDictOriginal['y'][dictSave['final_listPOS'][i]]), 0, 1)
            else:
                    pdf.cell(0, 10, 'Position of Matches: ', 0, 1)
                    pdf.cell(0, 10, 'X: ' + str(MouseDict['x'][dictSave['final_listPOS'][i]]), 0, 1)
                    pdf.cell(0, 10, 'Y:  ' + str(MouseDict['y'][dictSave['final_listPOS'][i]]), 0, 1)
            pdf.ln(h=20)

        pdf.output('MatchPDF.pdf', 'F')
    #
    clickValue=clicks
    return clicks
    # return json.dumps(clickValue)
    # print(clicks)

@app.callback(
    dash.dependencies.Output('hiddenDiv_PDFvalues', 'children'),
    [dash.dependencies.Input('hiddenDiv', 'children'),
     dash.dependencies.Input('timevar_graph_PP', 'figure'),
    dash.dependencies.Input('timevar_graph', 'figure'),
    dash.dependencies.Input('hiddenDiv_timevar', 'children'),
    dash.dependencies.Input('savePDF', 'n_clicks')
     ],
    [dash.dependencies.State('SCtext', 'value'),
     dash.dependencies.State('SCtext1', 'value'),
     dash.dependencies.State('SCtext2', 'value'),
     dash.dependencies.State('PreProcessing', 'value'),
      dash.dependencies.State('PreProcessing1', 'value'),
      dash.dependencies.State('PreProcessing2', 'value'),
     dash.dependencies.State('regex', 'value'),
     dash.dependencies.State('regex1', 'value'),
     dash.dependencies.State('regex2', 'value'),
    dash.dependencies.State('SCresult', 'value'),
    dash.dependencies.State('SCresult1', 'value'),
    dash.dependencies.State('SCresult2', 'value')
      ]
)
def calculatePDFvalues( matches, dataPP, data, timevar,click, SCtext1, SCtext2, SCtext3, PPtext1, PPtext2, PPtext3, regex1, regex2, regex3, SC1, SC2, SC3):

    matches=json.loads(matches)
    timevar=json.loads(timevar)
    matchInitial= matches['matchInitial']
    matchFinal=matches['matchFinal']

    matches_final=[]
    matchPercentage=[]
    SignalsX=[]
    SignalsY = []
    SignalsPP=[]
    NrofMatches=[]
    SCtext=[]
    SCtext.extend((SCtext1, SCtext2, SCtext3))
    PPtext = []
    PPtext.extend((PPtext1, PPtext2, PPtext3))
    regex=[]
    regex.extend((regex1,regex2,regex3))
    final_listPOS=[]
    SC=[]
    SC.extend((SC1, SC2, SC3))

    if len(matchInitial)!=0 and len(matchFinal)!=0:
        for j in range(len(timevar)):
            matches_intermediate = [] #lista com os indexes de todas as matches, para apenas 1 sinal de cada vez
            datax = np.array(data['data'][j]['x'])
            datay= np.array(data['data'][j]['y'])
            dataxPP = np.array(dataPP['data']['data'][j]['x'])
            datayPP = np.array(dataPP['data']['data'][j]['y'])
            NrofMatches.append(len(matchInitial[j]))
            for i in range(len(matchFinal[j])):  # cria uma lista com os indexes de todas as matches
                matches_intermediate.extend(range(matchInitial[j][i], matchFinal[j][i]))  # nao considera a matchFinal como match, seria preciso por +1 aqui

            matchPercentage.append((len(matches_intermediate)/len(dataxPP))*100)
            SignalsPP.append(datayPP)
            matches_final.append(matches_intermediate)
            SignalsX.append(datax)
            SignalsY.append(datay)


        final_listPOS=calculatePos(timevar, matches) #esta merda esta a dar erro, rever
    # print(final_listPOS)
    # for i in range(len(matches_final)):

    matches_final= list(filter(None, matches_final))
    # print('% match')
    # print(matchPercentage)
    # print(Signals)
    # print(SignalsPP)
    # print(NrofMatches)
    # print(matches_final)
    matchPercentage = list(filter(None, matchPercentage))
    SCtext = list(filter(None, SCtext))
    PPtext = list(filter(None, PPtext))
    regex = list(filter(None, regex))
    SC=list(filter(None, SC))

    # for i in range(len(matchPercentage)):
    #     Signals[i].tolist()
    #     SignalsPP[i].tolist()
    SignalsX= [l.tolist() for l in SignalsX]
    SignalsY=[l.tolist() for l in SignalsY]
    SignalsPP = [l.tolist() for l in SignalsPP]
    # if type(final_listPOS) is not list: #estava a dar erro para a timevar=straightness por isso pus esta exception
    final_listPOS= [l.tolist() for l in final_listPOS]
    # matches_final = [l.tolist() for l in matches_final]
    SignalsX=list(SignalsX)
    SignalsY = list(SignalsY)
    SignalsPP = list(SignalsPP)
    NrofMatches=list(NrofMatches)
    final_listPOS=list(final_listPOS)
    matches_final=list(matches_final)

    # print(matchPercentage)
    # print(matches_final)
    # print(matchPercentage)
    # print(Signals)

    dictSave={
        "NrofMatches":NrofMatches,
         "matchPercentage" : matchPercentage,
        "SignalsX": SignalsX,
        "SignalsY": SignalsY,
        "SignalsPP" : SignalsPP,
        "final_listPOS": final_listPOS,
        "matches_final" :matches_final,
        "timevar":timevar,
        "PPtext": PPtext,
        "SCtext" :SCtext,
        "regex" : regex,
        "SC": SC,
    }

    return json.dumps(dictSave, sort_keys=True)
#
# def set_default(obj):
#     if isinstance(obj, set):
#         return list(obj)
#     elif isinstance(obj, np.array):
#     print(obj)
#     print(type(obj))
#     raise TypeError

@app.callback(
    dash.dependencies.Output('text_spacevar', 'children'),
    [dash.dependencies.Input('showSpaceVar', 'n_clicks')])
def display_spacevar(n_clicks):
    if(n_clicks!= None):
        return '''OLA \n \
               LEL'''
     # str(round(min(space_var['l_strokes']),2))
        # * str(round(max(space_var['l_strokes']), 2)),
        # * str(round(min(space_var['straightness']), 2)),
        # * str(round(max(space_var['straightness']), 2)),
        # * str(round(space_var['jitter'],3)),
        # * str(round(min(space_var['angles']), 2)),
        # * str(round(max(space_var['angles']), 2)),
        # * str(round(min(space_var['w']), 2)),
        # * str(round(max(space_var['w']), 2)),
        # * str(round(min(space_var['curvatures']), 2)),
        # * str(round(max(space_var['curvatures']), 2))
        #
        # '''
            # "Length of strokes: [{0}, {1}] px/items/n" \
            #     "Straightness: [{0}, {1}] px/px/n " \
            #    "Jitter: [{4}]  /n" \
            #     "Angles: [{5}, {6}] /n" \
            #    "Angular Velocity (w): [{7}, {8}] /n"\
            #     "Curvature: [{9}, {10}] /n".format(
            #     str(round(min(space_var['l_strokes']),2)),
            #     str(round(max(space_var['l_strokes']), 2)),
            #     str(round(min(space_var['straightness']), 2)),
            #     str(round(max(space_var['straightness']), 2)),
            #     str(round(space_var['jitter'],3)),
            #     str(round(min(space_var['angles']), 2)),
            #     str(round(max(space_var['angles']), 2)),
            #     str(round(min(space_var['w']), 2)),
            #     str(round(max(space_var['w']), 2)),
            #     str(round(min(space_var['curvatures']), 2)),
            #     str(round(max(space_var['curvatures']), 2))
            #                                     )


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


def calculatePos(timevar, matches):
    global MouseDict
    matchInitial = matches['matchInitial']
    matchFinal = matches['matchFinal']
    nova_lista = [[], [], []]
    flat_list = [[], [], []]
    final_list=[]

    # for i in range(len(lista_matches)):
    #     valor_time=time_var['ttv'][lista_matches[i]]
    #     nova_lista.append(MouseDict['t'].index(time_var['ttv'][lista_matches[i]]))

    time_array = np.array(MouseDict['t'])
    listA = ["vt", "vx", "vy", "a", "jerk"]
    listB = ["xt", "yt"]
    listC = ['straight', 'lenStrokes', 'pausescumsum', 'time', 'clicks']

    # print(time_var['ttv'][matchInitial[0][0]])
    # print(time_var['ttv'][matchFinal[0][0]])
    # print(timevar)
    if np.size(matchInitial)>0:
        for j in range(len(timevar)):  # para iterar entre os varios sinais


            if timevar[j] in listA:

                for i in range(len(matchInitial[j])):  # para iterar dentro do mm sinal entre as varias matches
                    if i == len(matchInitial[j]) - 1:
                        nova_lista[j].append(np.where((time_array >= time_var['ttv'][matchInitial[j][i]]) & (time_array <= time_var['ttv'][matchFinal[j][i] - 1])))
                    else:
                        nova_lista[j].append(np.where((time_array >= time_var['ttv'][matchInitial[j][i]]) & (time_array <= time_var['ttv'][matchFinal[j][i]])))

            if (timevar[j] in listB):

                for i in range(len(matchInitial[j])):  # esta a sair fora do sinal qd a match é o ultimo ponto, acho que nao esta correcto assim
                    if i == len(matchInitial[j]) - 1:
                        nova_lista[j].append(np.where((time_array >= time_var['tt'][matchInitial[j][i]]) & (time_array <= time_var['tt'][matchFinal[j][i] - 1])))
                    else:
                        nova_lista[j].append(np.where((time_array >= time_var['tt'][matchInitial[j][i]]) & (time_array <= time_var['tt'][matchFinal[j][i]])))  # pus -1 porque estava a sair fora do vector

            if (timevar[j] in listC):
                for i in range(len(matchInitial[j])):
                    flat_list[j].extend(range(matchInitial[j][i], matchFinal[j][i]))

            # print('noval')
            # print(nova_lista)
            if (len(nova_lista[j]) != 0):  # para fazer flat das matches, acho que se usar extend em vez de append isto fica desnecessario
                flat_list[j] = np.concatenate(nova_lista[j], axis=1)[0]
                # print(flat_list)

        flat_list = np.array(flat_list)
        final_list = []

        for i in range(len(flat_list)):  # para remover a parte vazia dos vectores pq o filter nao funciona
            if len(flat_list[i]) > 0:
                final_list.append(flat_list[i])


    return final_list


if __name__ == '__main__':
    print(dcc.__version__)
    app.run_server()

# @app.callback( funcçao do radio com o slider que tive de tirar porque nao funcionava
#     dash.dependencies.Output('timevar_graph', 'figure'),
#     [dash.dependencies.Input('Radio', 'value'),
#      dash.dependencies.Input('slider','value')])
# def update_figure(selected_option, value):
#     # value=int(value)
#
#     if(str(selected_option) in ("vt, vx, vy")):
#         traces=[]
#         traces.append(go.Scatter(
#                 x=time_var['ttv'],
#                 y=time_var[str(selected_option)],
#                 text=selected_option[0],
#                 opacity=0.7,
#                 # marker={
#                 #     'size': 5,
#                 #     'line': {'width': 0.5, 'color': 'white'}
#                 # },
#                 line= {'width': 2, 'color': 'black'},
#                 name=str(selected_option)
#             ))
#     elif(str(selected_option) in ("xt, yt, a, jerk")):
#         traces = []
#         traces.append(go.Scatter(
#             x=time_var['tt'],
#             y=time_var[str(selected_option)],
#             text=selected_option[0],
#             opacity=0.7,
#             # marker={
#             #     'size': 5,
#             #     'line': {'width': 0.5, 'color': 'white'}
#             # },
#             line={'width': 2, 'color': 'black'},
#             name=str(selected_option)
#         ))
#
#     traces.append(go.Scatter(
#         x=[time_var['ttv'][value]],
#         y=[time_var[str(selected_option)][value]],
#         mode='markers',
#         name='sliderpos'))
#
#
#         # print(len(time_var['jerk']))
#         # print(len(time_var['xt']))
#         # filtered_df = dM[selected_option]
#     return {
#         'data':traces,
#         'layout': go.Layout(
#             xaxis={'title': 'Time'},
#             yaxis={'title': selected_option},
#             legend={'x': 0, 'y': 1},
#             hovermode='closest'
#         )
#
#     }
def PP_parser_tab(parse):

    CutString = parse.split()
    for i in range(len(CutString)):  # estava range(len(CutString)-1 for some reason antes
        if (CutString[i] == 'H'):
            # Function Amplitude
            data['data'][0]['y'] = highpass(data['data'][0]['y'], int(CutString[i + 1]))  # aplicar o filtro aos dados y

        elif (CutString[i] == 'L'):
            # Function Low Pass
            data['data'][0]['y'] = lowpass(data['data'][0]['y'], int(CutString[i + 1]))

        elif (CutString[i] == 'BP'):
            # Function Band Pass
            data['data'][0]['y'] = bandpass(data['data'][0]['y'], int(CutString[i + 1]), int(CutString[i + 2]))

        elif (CutString[i] == 'S'):
            # Function Smooth
            smooth_signal = smooth(np.array(data['data'][0]['y'])), int(CutString[ i + 1])  # estranho porque o smooth retorna uma lista com os dados na 1ªpos e o valor do smooth na segunda
            data['data'][0]['y'] = smooth_signal[0]

        elif (CutString[i] == 'ABS'):
            # Function Absolute
            data['data'][0]['y'] = np.absolute(data['data'][0]['y'])

    return data

