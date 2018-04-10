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
from SymbolicMethods import DiffC, Diff2C, RiseAmp, AmpC
from AuxiliaryMethods import _plot, detect_peaks,merge_chars
import base64
import io
import dash_table_experiments as dt
import datetime

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

def DrawShapes(matchInitial, matchFinal, datax, datay):
    shapes=[]
    for i in range(np.size(matchInitial)):
        shape={
            'type': 'rect',
            # x-reference is assigned to the x-values
            'xref': 'x',
            # y-reference is assigned to the plot paper [0,1]
            'yref': 'y',
            'x0': datax[matchInitial[i]-1],
            'y0': min(datay),
            'x1': datax[matchFinal[i]-1],
            'y1': max(datay),
            'fillcolor': '#A7CCED',
            'opacity': 0.7,
            'line': {
                'width': 0,
                }}
        shapes.append(shape)

    return shapes


def UpdateTimeVarGraph(traces, selected_option):
    global time_var
    if (str(selected_option) in ("vt, vx, vy, a, jerk")):

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
    elif (str(selected_option) in ("xt, yt")):
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

    return traces

def createLayoutTimevar(value):
    Titles=dict(vt='Velocity in time',
         vy='Velocity in Y',
         vx='Velocity in X',
         jerk='Jerk',
         a='Acceleration in time',
         xt='X position in time',
         yt='Y position in time'
         )
    layout=go.Layout(
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
        # {'label': 'Test', 'value': 'test'}
    ],
    values=['XY'],
    style={'display':'inline-block', 'width':'100%'})

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

Div_SC = dcc.Graph(id='regexgraph', style={'display': 'none', 'width':'100%'})

Div_S = dcc.Graph(id='regexgraph', style={'display': 'none', 'width':'100%' })

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
                    html.Button('ABS', id='absolute', style=styleB, title='Module')]
                ),
                html.Div(
                    [dcc.Input(id='PreProcessing',
                               placeholder='Ex: "H 50 L 10"',
                               type='text',
                               value=''),
                    html.Button('Pre-Processing', id='preprocess', style=styleB)]
                ),
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
                    html.Button('A', id='Amp', value='', style=styleB, title='Amplitude'),
                    html.Button('1D', id='diff1', style=styleB, title='1st Derivative'),
                    html.Button('2D', id='diff2', style=styleB, title='2nd Derivative'),
                    html.Button('RA', id='riseamp', style=styleB, title='RiseAmp')]
                ),
                html.Div(dcc.Input(id='SCtext',
                    placeholder='Enter Symbolic Methods',
                    type='text',
                    value='',
                    # style={'display': 'none'}
                )),
                html.Div(
                    html.Button('Symbolic Connotation', id='SCbutton', style=styleB)),
                html.Div(
                    dcc.Textarea(
                    id="SCtest",
                    placeholder='Your symbolic series will appear here.',
                    value='',
                    # style={'margin-bottom':'7px'}
                    )
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

                html.Div([
                        dcc.Input(id='regex',
                        placeholder='Enter Regular Expression...',
                        type='text',
                        value='',
                        # style={'display': 'none'}
                  ),
                html.Button('Search Regex', id='searchregex', style=styleB)
                        ],
                         style={
                            'width': '25%',
                            'fontFamily': 'Sans-Serif',
                            'float' : 'left',
                            'backgroundColor': colors['background']}


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
                {'label' : 'Acceleration', 'value': 'a'}
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
    html.Div([
        html.Button('Show Space Vars', id='showSpaceVar'),
        dcc.Markdown(id ='text_spacevar')]


    )
])

#------------------------------------------------------
#   Callback Functions
#-----------------------------------------------------
def createDictionary(df):
    global time_var, space_var
    global MouseDict
    MouseX=[]
    MouseY = []
    MouseTime = []
    vars={}

    print(len(df[3]))
    MouseX=df[3][::7]

    # x = [d for d in x if re.match('\d+', d)]  # procura na lista os que sao numeros, para retirar os undefined
    # MouseX = np.array(x).astype(int)  # converte string para int, so é preciso isto se o ficheiro tiver undefineds


    MouseY=df[4][::7]
    # y = [d for d in y if re.match('\d+', d)]  # procura na lista os que sao numeros, para retirar os undefined
    # MouseY= np.array(y).astype(int)
    # print(MouseY)
    for i in range(len(df[11])):
        if i==0:
            initial_time=df[11][0]/1000
            MouseTime.append(0)
        else:
            MouseTime.append((df[11][i]/1000)-initial_time)
    MouseTime=MouseTime[::7]

    # print(numpy.isnan(MouseX).any())

    # mynewlist = [s for s in MouseX if s.isdigit()]

    print(len(MouseX))
    # print(len(mynewlist))

    MouseDict = dict(t=MouseTime, x=MouseX, y=MouseY)
    dM = pd.DataFrame.from_dict(MouseDict)
    time_var, space_var = interpolate_data(dM, t_abandon=20)
    vars = {'time_var': time_var, 'space_var': space_var}
    # print(vars)


def parse_contents(contents, filename, date):
    # print(contents)
    content_type, content_string = contents.split(',')
    # print(content_type)
    # print(content_string)
    decoded = base64.b64decode(content_string)
    # print(decoded)
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
    dash.dependencies.Output('hiddenDiv_timevar', 'children'),
    [dash.dependencies.Input('dropdown_timevar', 'value')]
)
def updateTimeVar(value):
    return json.dumps(value)

@app.callback(
        dash.dependencies.Output('hiddenDiv', 'children'),
        [dash.dependencies.Input('regex', 'value'),
         dash.dependencies.Input('SCtest', 'value'),
         dash.dependencies.Input('searchregex', 'n_clicks'),
         ]
    )
def updateHiddenDiv(regex, string, n_clicks):
    global lastclick2
    matches={"matchInitial":[], "matchFinal":[]}
    if (n_clicks != None):
        if (n_clicks > lastclick2 and len(string)>0):
            matchInitial = []
            matchFinal = []
            regit = re.finditer(regex, string)

            for i in regit:
                matchInitial.append((int(i.span()[0])))
                matchFinal.append(int(i.span()[1]))

            matchInitial = np.array(matchInitial)
            matchFinal = np.array(matchFinal)
            matches['matchInitial']=matchInitial.tolist()
            matches['matchFinal']= matchFinal.tolist()


    return json.dumps(matches, sort_keys=True)


@app.callback(
    dash.dependencies.Output('regexgraph', 'figure'),
    [dash.dependencies.Input('regex', 'value'),
     dash.dependencies.Input('SCtest', 'value'),
     dash.dependencies.Input('searchregex', 'n_clicks'),
     dash.dependencies.Input('timevar_graph_PP', 'figure')
     ]
)
def RegexParser(regex, string, n_clicks, data):
    global lastclick2

    if (n_clicks != None):
        if(n_clicks>lastclick2 and len(string)>0):
            str=numpy.asarray(string)
            matches = []
            traces=[]

            matchInitial=[]
            matchFinal = []

            regit = re.finditer(regex,string)

            for i in regit:
                matchInitial.append((int(i.span()[0])))
                matchFinal.append(int(i.span()[1]))


            matchInitial=np.array(matchInitial)
            matchFinal=np.array(matchFinal)
            datax = np.array(data['data']['data'][0]['x'])
            datay= np.array(data['data']['data'][0]['y'])
            if (len(matchInitial)>0 and len(matchFinal)>0 ):
                traces.append(go.Scatter( #match initial append
                    x=datax[matchInitial-1],
                    y=datay[matchInitial-1],
                    # text=selected_option[0],
                    mode='markers',
                    opacity=0.7,
                    marker=dict(
                        size=10,
                        color='#EAEBED',
                        line=dict(
                            width=2)

                    ),
                    name='Match Initial'

                ))
                traces.append(go.Scatter( #match final append
                    x=datax[matchFinal-1], #porque -1 ?
                    y=datay[matchFinal-1],
                    # text=selected_option[0],
                    mode='markers',
                    opacity=0.7,
                    marker=dict(
                        size=10,
                        color='#006989',
                        line=dict(
                            width=2)

                    ),
                    name='Match Final'
                ))
            traces.append(go.Scatter(
                x=datax,
                y=datay,
                mode='lines',
                line=dict(
                    color='black',
                    width=0.7

                ),
                name='Signal'
            ))

            return {
                'data': traces,
                'layout': go.Layout(
                    hovermode='closest',
                    shapes =DrawShapes(matchInitial,matchFinal,datax, datay),
                    autosize=True,
                    xaxis=dict(ticks='', showgrid=False, zeroline=False),
                    yaxis=dict(ticks='', showgrid=False, zeroline=False)
                )}

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
    dash.dependencies.Output('SCtext', 'value'),[
    dash.dependencies.Input('Amp', 'n_clicks'),
    dash.dependencies.Input('diff1', 'n_clicks'),
    dash.dependencies.Input('diff2', 'n_clicks'),
    dash.dependencies.Input('riseamp', 'n_clicks')],
    [dash.dependencies.State('SCtext', 'value')]
)
def SymbolicConnotationWrite(a1,a2,a3,a4, finalStr):
    global  preva1, preva2, preva3, preva4 #banhada com variaveis globais para funcionar, convem mudar
    if(a1!= None): # tem o problema de nao limpar, se calhar precisa de um botao para limpar
        if(a1>preva1): #Amplitude
            finalStr+= 'A '
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
    return str(finalStr)

#Nota: input dos dados processados e nao originais
@app.callback(
    dash.dependencies.Output('SCtest', 'value'),
    [dash.dependencies.Input('SCbutton', 'n_clicks'),
    dash.dependencies.Input('SCtext', 'value'),
    dash.dependencies.Input('timevar_graph_PP', 'figure')]
)
def SymbolicConnotationStringParser(n_clicks, parse, data):
    global lastclick1
    finalString=[]


    if (n_clicks != None):

        if(n_clicks>lastclick1): # para fazer com que o graf so se altere quando clicamos no botao e nao devido aos outros callbacks-grafico ou input box
            CutString=parse.split()
            for i in range(len(CutString)-1):
                if(CutString[i] =='A'):
                    #function Amp
                    finalString.append(AmpC(data['data']['data'][0]['y'],float(CutString[i+1])))

                elif(CutString[i] =='1D'):
                    #Function 1st Derivative
                    finalString.append( DiffC(data['data']['data'][0]['y'], float(CutString[i + 1])))

                elif (CutString[i] == '2D'):
                    #Function 2nd Derivative
                    finalString.append( Diff2C(data['data']['data'][0]['y'], float(CutString[i + 1])))

                elif (CutString[i] == 'R'):
                    #Function RiseAmp
                    finalString.append(RiseAmp(data['data']['data'][0]['y'], float(CutString[i + 1]))) #nao sei se esta a fazer bem



            finalString=merge_chars(np.array(finalString))  # esta a dar um erro estranho quando escrevo a caixa--FIXED
        lastclick1 = n_clicks

    return finalString



@app.callback(
    dash.dependencies.Output('timevar_graph_PP', 'figure'),
    [dash.dependencies.Input('preprocess', 'n_clicks'),
    dash.dependencies.Input('timevar_graph', 'figure')],
    [dash.dependencies.State('PreProcessing', 'value')]
)
def PreProcessStringParser(n_clicks, data, parse):

    global lastclick
    if (n_clicks != None):
        if(n_clicks>lastclick): # para fazer com que o graf so se altere quando clicamos no botao e nao devido aos outros callbacks-grafico ou input box

            # maxTime=max(data['data'][0]['x'])
            # lengthOfSeries=len(data['data'][0]['x'])
            # fs=int(1/(maxTime/lengthOfSeries))

            fs=[]
            for i in range(len(data['data'][0]['x'])-1):
                fs.append(1/(data['data'][0]['x'][1]-data['data'][0]['x'][0]))
            fs=np.mean(fs)
            CutString=parse.split()
            for i in range(len(CutString)): #estava range(len(CutString)-1 for some reason antes
                if(CutString[i] =='H'):
                    #Function Amplitude
                    data['data'][0]['y']=highpass(data['data'][0]['y'],int(CutString[i+1]), fs=fs) #aplicar o filtro aos dados y

                elif(CutString[i] =='L'):
                    #Function Low Pass
                    data['data'][0]['y']=lowpass(data['data'][0]['y'], int(CutString[i+1]), fs=fs)

                elif (CutString[i] == 'BP'):
                    #Function Band Pass
                    data['data'][0]['y']=bandpass(data['data'][0]['y'], int(CutString[i+1]), int(CutString[i+2]), fs=fs)

                elif (CutString[i] == 'S'):
                    #Function Smooth
                    smooth_signal = smooth(np.array(data['data'][0]['y']), int(CutString[i+1])) #estranho porque o smooth retorna uma lista com os dados na 1ªpos e o valor do smooth na segunda
                    data['data'][0]['y']=smooth_signal

                elif (CutString[i] == 'ABS'):
                    #Function Absolute
                    data['data'][0]['y']=np.absolute(data['data'][0]['y'])



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
    dash.dependencies.Output('timevar_graph', 'figure'),
    [dash.dependencies.Input('dropdown_timevar', 'value')])
def update_timevarfigure(selected_option):

    traces=[]
    layout={}

    if(len(selected_option) == 1):

        traces = UpdateTimeVarGraph(traces, selected_option[0])
        layout= createLayoutTimevar(selected_option[0])

    if (len(selected_option)>1):
         for i in range(np.size(selected_option)):
            traces = UpdateTimeVarGraph(traces, selected_option[i])
         layout=createLayoutTimevar(selected_option[0])

    return {
        'data':traces,
        'layout': layout
    }

@app.callback(
    dash.dependencies.Output('PosGraph', 'figure'),
    # [dash.dependencies.Input('interpolate', 'n_clicks'),
     [dash.dependencies.Input('checklistheatmap', 'values'),
     dash.dependencies.Input('hiddenDiv', 'children'),
     dash.dependencies.Input('hiddenDiv_timevar', 'children')
     # dash.dependencies.Input('sliderpos', 'value')
     ])
def interpolate_graf(value, json_data, timevar):
    global MouseDict
    global space_var
    matches = json.loads(json_data)
    timevar=json.loads(timevar)

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
                )

            ))
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

        if(value[i]=='interpolate'):
            traces.append(go.Scatter(
                x=space_var['xs'],
                y=space_var['ys'],
                # text=selected_option[0],
                opacity=0.7,
                # marker={
                #     'size': 5,
                #     'line': {'width': 0.5, 'color': 'white'}
                # },
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





    if(np.size(matches['matchInitial'])>0):

        lista_matches=[]

        for i in range(len(matchInitial)):
            for a in range(matchInitial[i], matchFinal[i]+1): #+1 porque o range faz ate o valor-1
                lista_matches.append(a)

        nova_lista=[]

        # for i in range(len(lista_matches)):
        #     valor_time=time_var['ttv'][lista_matches[i]]
        #     nova_lista.append(MouseDict['t'].index(time_var['ttv'][lista_matches[i]]))

        time_array=np.array(MouseDict['t'])
        listA=["vt", "vx", "vy", "a", "jerk"]
        listB=["xt", "yt"]

        if timevar[0] in listA: #ainda so funciona para 1 signal
            for i in range (len(matchInitial)):
                nova_lista.append(np.where( (time_array>= time_var['ttv'][matchInitial[i]]) & (time_array<= time_var['ttv'][matchFinal[i]]) ))

        if(timevar[0] in listB):
            for i in range (len(matchInitial)):
                nova_lista.append(np.where( (time_array>= time_var['tt'][matchInitial[i]]) & (time_array<= time_var['tt'][matchFinal[i]]) ))


        flat_list=np.concatenate(nova_lista, axis=1)[0]
        flat_list=np.array(flat_list)


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

        Xpos=np.array(MouseDict['x'])
        Ypos=np.array(MouseDict['y'])
        traces.append(go.Scatter(
            x=Xpos[flat_list],
            y=Ypos[flat_list],
            # text=selected_option[0],
            opacity=1,
            mode='markers',
            # marker={
            #     'size': 5,
            #     'line': {'width': 0.5, 'color': 'white'}
            # },
            marker={'size': 5, 'color': '#A7CCED'},
            name="Matches"
        ))

    return {
            'data': traces,
            'layout': layout
    }


@app.callback(dash.dependencies.Output('text_spacevar', 'children'),
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

