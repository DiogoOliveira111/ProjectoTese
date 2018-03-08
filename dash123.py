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
from scipy import signal
from scipy.signal import filtfilt


#Extra Functions

# Filtering methods
def smooth(input_signal, window_len=10, window='hanning'):
    """
    @brief: Smooth the data using a window with requested size.
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the beginning and end part of the output signal.
    @param: input_signal: array-like
                the input signal
            window_len: int
                the dimension of the smoothing window. the default is 10.
            window: string.
                the type of window from 'flat', 'hanning', 'hamming',
                'bartlett', 'blackman'. flat window will produce a moving
                average smoothing. the default is 'hanning'.
    @return: signal_filt: array-like
                the smoothed signal.
    @example:
                time = linspace(-2,2,0.1)
                input_signal = sin(t)+randn(len(t))*0.1
                signal_filt = smooth(x)
    @see also:  numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman,
                numpy.convolve, scipy.signal.lfilter
    @todo: the window parameter could be the window itself if an array instead
    of a string
    @bug: if window_len is equal to the size of the signal the returning
    signal is smaller.
    """

    if input_signal.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if input_signal.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return input_signal

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("""Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'""")

    sig = np.r_[2 * input_signal[0] - input_signal[window_len:0:-1],
                input_signal,
                2 * input_signal[-1] - input_signal[-2:-window_len - 2:-1]]

    if window == 'flat':  # moving average
        win = np.ones(window_len, 'd')
    else:
        win = eval('np.' + window + '(window_len)')

    sig_conv = np.convolve(win / win.sum(), sig, mode='same')

    return sig_conv[window_len: -window_len]



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

# Auxiliary methods
def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):
    """Detect peaks in data based on their amplitude and other features.
    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).
    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.
    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`
    The function can handle NaN's
    See this IPython Notebook [1]_.
    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb"""

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan - 1, indnan + 1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                              & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind

# Connotation Methods
def AmpC(s, t, p='>'):

    thr = (float(np.max(s) - np.min(s)) * t) + np.min(s)
    if (p == '<'):
        s1 = (s <= (thr)) * 1
    elif (p == '>'):
        s1 = (s >= (thr)) * 1

    return s1

def _plot(x, mph, mpd, threshold, edge, valley, ax, ind):
    """Plot results of the detect_peaks function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(x, 'b', lw=1)
        if ind.size:
            label = 'valley' if valley else 'peak'
            label = label + 's' if ind.size > 1 else label
            ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                    label='%d %s' % (ind.size, label))
            ax.legend(loc='best', framealpha=.5, numpoints=1)
        ax.set_xlim(-.02 * x.size, x.size * 1.02 - 1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)
        ax.set_xlabel('Data #', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        mode = 'Valley detection' if valley else 'Peak detection'
        ax.set_title("%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"
                     % (mode, str(mph), mpd, str(threshold), edge))


def DiffC(s, t, signs=['-', '_', '+']):
    # Quantization of the derivative.
    # TODO: Implement a better way of selecting chars
    ds1 = np.diff(s)
    x = np.empty(len(s), dtype=str)
    thr = (np.max(ds1) - np.min(ds1)) * t
    x[np.where(ds1 <= -thr)[0]] = signs[0]
    x[np.where(np.all([ds1 <= thr, ds1 >= -thr], axis=0))[0]] = signs[1]
    x[np.where(thr <= ds1)[0]] = signs[2]
    x[-1] = x[-2]

    return x


def Diff2C(s, t, symbols=['-', '_', '+']):
    # Quantization of the derivative.
    # TODO: Implement a better threshold methodology.
    dds1 = np.diff(np.diff(s))
    x = np.empty(len(s), dtype=str)
    thr = (np.max(dds1) - np.min(dds1)) * t
    x[np.where(dds1 <= -thr)[0]] = symbols[0]
    x[np.where(np.all([dds1 <= thr, dds1 >= -thr], axis=0))[0]] = symbols[1]
    x[np.where(thr <= dds1)[0]] = symbols[2]
    x[-1] = x[-2]

    return x


def RiseAmp(Signal, t):
    # detect all valleys
    val = detect_peaks(Signal, valley=True)
    # final pks array
    pks = []
    thr = ((np.max(Signal) - np.min(Signal)) * t) + np.mean(Signal)
    # array of amplitude of rising with size of signal
    risingH = np.zeros(len(Signal))
    Rise = np.array([])

    for i in range(0, len(val) - 1):
        # piece of signal between two successive valleys
        wind = Signal[val[i]:val[i + 1]]
        # find peak between two minimums
        pk = detect_peaks(wind, mph=0.1 * max(Signal))
        # print(pk)
        # if peak is found:
        if (len(pk) > 0):
            # append peak position
            pks.append(val[i] + pk)
            # calculate rising amplitude
            # val=np.array(val)
            # pk=np.array(pk)
            # risingH=np.array(risingH)
            wind=np.array(wind)
            risingH[val[i]:val[i + 1]] = [wind[pk] - Signal[val[i]] for a in range(val[i], val[i + 1])]
            Rise = np.append(Rise, (wind[pk] - Signal[val[i]]) > thr)

    risingH = np.array(risingH > thr).astype(int)
    Rise = Rise.astype(int)

    return risingH

def merge_chars(string_matrix):
    """
    Function performs the merge of the strings generated with each method. The function assumes
    that each string is organized in the StringMatrix argument as a column.
    The function returns the merged string.
    """

    col = np.size(string_matrix, axis=0)
    lines = np.size(string_matrix, axis=1)

    Str = ""
    for l in range(0, lines):
        for c in range(0, col):
            Str += str(string_matrix[c][l])

    return Str

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
# app.scripts.config.serve_locally = True #no idea what this does

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
html.Div([
    dcc.Tabs(
        tabs=[
            {'label': 'Mouse Movement', 'value': 'XY'},
            {'label': 'Pre-Processing', 'value': 'PP'},
            {'label': 'Symbolic Connotation', 'value': 'SC'},
            {'label': 'Search', 'value': 'S'}
        ],
        value='XY',
        id='tabs'
    ),
    html.Div(id='tab-output')
], style={
    'width': '80%',
    'fontFamily': 'Sans-Serif',
    'margin-left': 'auto',
    'margin-right': 'auto'
}),
html.Div(children= dcc.Graph(
        id='PosGraf',
        style={'display': 'none' #inline-block
        },
        figure={
            'data': [
                go.Scatter(
                    y = MouseDict['y'],
                    x= MouseDict['x'],
                    mode= 'markers',
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
                xaxis={ 'title': 'X position'},
                yaxis={'title': 'y position'},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                legend={'x': 0, 'y': 1},
                hovermode='closest')

        }

    )),
html.Div([
html.Button('Interpolate', id='interpolate',
            style={'display': 'none'  # para nao mostrar, so com o tab certo
                   }
),
dcc.Slider( id='slider',
            # style={'display': 'none'  # para nao mostrar, so com o tab certo
            #        },
    min=0,
    max=len(time_var['ttv']),
    step=1,
    value=0,
)]),

html.Div(
    dcc.RadioItems(id='Radioheatmap',
                    style={'display': 'none'  # para nao mostrar, so com o tab certo
                   },
       options=[
           {'label': 'Positional Map', 'value': 'xy'},
           {'label': 'Heatmap', 'value': 'heat'},
       ], value='xy'

    )),


    html.Div([
    dcc.Graph(id='timevar_graph',
              style={'display': 'none'  # inline-block
                     }),
    dcc.RadioItems(
        id='Radio',
        style={'display': 'none'},

       options=[
           {'label': 'Velocity in X', 'value': 'vx'},
           {'label': 'Velocity in Y', 'value': 'vy'},
           {'label': 'Jerk', 'value': 'jerk'},
           {'label': 'X position in t', 'value': 'xt'},
           {'label': 'Y Position in t', 'value': 'yt'},
           {'label' : 'Velocity', 'value': 'vt'},
            {'label' : 'Acceleration', 'value': 'a'}
       ], value='vt'

    ),

    html.Div(id='output_spacevar'), #falta esconder
             dcc.Markdown(id='text_spacevar')

]),
    html.Div([
        html.Button('HighPass (H)', id='highpass', value='', style={'display': 'none'}),
        html.Button('LowPass (L)', id='lowpass', style={'display': 'none'}),
        html.Button('BandPass (Bp)', id='bandpass', style={'display': 'none'}),
        html.Button('Smooth (S)', id='smooth', style={'display': 'none'}),
        html.Button('Module (Abs)', id='absolute', style={'display': 'none'})]
),
    html.Div(dcc.Input(id='PreProcessing',
    placeholder='Ex: "H 50 L 10"',
    type='text',
    value='',
    style={'display': 'none'}
)),
    html.Div([
        html.Button('Pre-Processing', id='preprocess', style={'display': 'none'}),
        dcc.Graph(id='timevar_graph_PP', style={'display': 'none'})
    ]),

    html.Div([
        html.Button('Amplitude (A)', id='Amp', value='', style={'display': 'none'}),
        html.Button('1st Derivative (D)', id='diff1', style={'display': 'none'}),
        html.Button('2nd Derivative (C)', id='diff2', style={'display': 'none'}),
        html.Button('RiseAmp (R)', id='riseamp', style={'display': 'none'})]
),
    html.Div(dcc.Input(id='SCtext',
    placeholder='Enter a value...',
    type='text',
    value='',
    style={'display': 'none'}
)),
    html.Div([
        html.Button('Symbolic Connotation', id='SCbutton',style={'display': 'none'} ),
    dcc.Textarea(
        id="SCtest",
        placeholder='Enter a value...',
        value='This is a TextArea component',
        style={'display': 'none'}
    )
    ]),

    html.Div([
        dcc.Input(id='regex',
        placeholder='Enter Regular Expression...',
        type='text',
        value='',
        style={'display': 'none'}),
        html.Button('Search Regex', id='searchregex', style={'display': 'none'}),
        dcc.Graph(id='regexgraph', style={'display': 'none'})
    ])


])

@app.callback(
    dash.dependencies.Output('tab-output', 'children'),
    [dash.dependencies.Input('tabs', 'value')])
def display_content(value):
    if value=='XY':
        return html.Div([
            dcc.Graph(
                id='PosGraf',
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

                       }),
            html.Button('Interpolate', id='interpolate'),

                dcc.RadioItems(id='Radioheatmap',
                               options=[
                                   {'label': 'Positional Map', 'value': 'xy'},
                                   {'label': 'Heatmap', 'value': 'heat'},
                               ], value='xy'

                               ),
                ])
    if value=='PP':
        return html.Div([
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
                   ], value='vt'

                ),

                html.Div(id='output_spacevar'),
                dcc.Markdown(id='text_spacevar')  ,
            html.Div([
                html.Button('HighPass (H)', id='highpass', value='' ),
                html.Button('LowPass (L)', id='lowpass'),
                html.Button('BandPass (Bp)', id='bandpass'),
                html.Button('Smooth (S)', id='smooth'),
                html.Button('Module (Abs)', id='absolute')]
            ),
            html.Div(dcc.Input(id='PreProcessing',
                               placeholder='Ex: "H 50 L 10"',
                               type='text',
                               value='',
                               )),
            html.Div([
                html.Button('Pre-Processing', id='preprocess'),
                dcc.Graph(id='timevar_graph_PP')
            ])
        ])
    if value=='SC':
        return html.Div([dcc.Graph(id='timevar_graph_PP'),
                         html.Div([
                             html.Button('Amplitude (A)', id='Amp', value=''),
                             html.Button('1st Derivative (D)', id='diff1'),
                             html.Button('2nd Derivative (C)', id='diff2'),
                             html.Button('RiseAmp (R)', id='riseamp')]
                         ),
                         html.Div(dcc.Input(id='SCtext',
                                            placeholder='Enter a value...',
                                            type='text',
                                            value='',
                                            )),
                         html.Div(
                             html.Button('Symbolic Connotation', id='SCbutton')),
                         html.Div(
                             dcc.Textarea(
                                 id="SCtest",
                                 placeholder='Enter a value...',
                                 value='Your Symbolic Time Series will appear here',
                                 style={'width': '100%'})
                                 )

                         ])
    if value=='S':
        return html.Div([
                dcc.Input(id='regex',
                          placeholder='Enter Regular Expression...',
                          type='text',
                          value=''),
                html.Button('Search Regex', id='searchregex'),
                dcc.Graph(id='regexgraph')
            ])


@app.callback(
    dash.dependencies.Output('regexgraph', 'figure'),
    [dash.dependencies.Input('regex', 'value'),
     dash.dependencies.Input('SCtest', 'value'),
     dash.dependencies.Input('searchregex', 'n_clicks'),
     dash.dependencies.Input('timevar_graph_PP', 'figure')]
)
def RegexParser(regex, string, n_clicks, data):
    global lastclick2
    if (n_clicks != None):
        if(n_clicks>lastclick2):
            str=numpy.asarray(string)
            matches = []
            traces=[]
            for i in range(len(string)):
                matchInitial=[]
                matchFinal = []
                regit = re.finditer(regex,string)
                for i in regit:
                    matchInitial.append((int(i.span()[0])))
                    matchFinal.append(int(i.span()[1]))
                    # [match.append((int(i.span()[0]),
                #                int(i.span()[1]))) for i in regit]
                # print(data)

            matchInitial=np.array(matchInitial)
            matchFinal=np.array(matchFinal)
            datax = np.array(data['data'][0]['x'])
            datay=np.array(data['data'][0]['y'])
            # print(datax[matchInitial])
            # print(datay[matchInitial])

            traces.append(go.Scatter( #match initial append
                x=datax[matchInitial],
                y=datay[matchInitial],
                # text=selected_option[0],
                mode='markers',
                opacity=0.7,
                marker=dict(
                    size=10,
                    color='green',
                    line=dict(
                        width=2)

                ),
                name='Match Initial'
                # opacity=1,
                #
                # marker={
                #     'size': 10,
                #     'line': {'width': 0.5, 'color': 'white'}
                # },

            ))
            traces.append(go.Scatter( #match final append
                x=datax[matchFinal],
                y=datay[matchFinal],
                # text=selected_option[0],
                mode='markers',
                opacity=0.7,
                marker=dict(
                    size=10,
                    color='red',
                    line=dict(
                        width=2)

                ),
                name='Match Final'
            ))
            traces.append(go.Scatter(
                x=data['data'][0]['x'],
                y=data['data'][0]['y'],
                mode='lines',
                line=dict(
                    color='black',
                    width=1

                ),
                name='Signal'
            ))
            return {
                'data': traces,
                'layout': go.Layout(
                    legend={'x': 0, 'y': 1},
                    hovermode='closest',

                )}



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

@app.callback(
    dash.dependencies.Output('SCtest', 'value'),
    [dash.dependencies.Input('SCbutton', 'n_clicks'),
    dash.dependencies.Input('SCtext', 'value'),
    dash.dependencies.Input('timevar_graph', 'figure')]
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
                    finalString.append(AmpC(data['data'][0]['y'],float(CutString[i+1])))

                elif(CutString[i] =='1D'):
                    #Function 1st Derivative
                    finalString.append( DiffC(data['data'][0]['y'], float(CutString[i + 1])))

                elif (CutString[i] == '2D'):
                    #Function 2nd Derivative
                    finalString.append( Diff2C(data['data'][0]['y'], float(CutString[i + 1])))

                elif (CutString[i] == 'R'):
                    #Function RiseAmp
                    finalString.append(RiseAmp(data['data'][0]['y'], float(CutString[i + 1]))) #nao sei se esta a fazer bem


        lastclick1=n_clicks
        print(finalString)
        finalString=merge_chars(np.array(finalString))  # esta a dar um erro estranho quando escrevo a caixa
        return finalString


@app.callback(
    dash.dependencies.Output('timevar_graph_PP', 'figure'),
    [dash.dependencies.Input('preprocess', 'n_clicks'),
    dash.dependencies.Input('PreProcessing', 'value'),
    dash.dependencies.Input('timevar_graph', 'figure')]
)

def PreProcessStringParser(n_clicks, parse, data):
    global lastclick
    if (n_clicks != None):
        if(n_clicks>lastclick): # para fazer com que o graf so se altere quando clicamos no botao e nao devido aos outros callbacks-grafico ou input box
            CutString=parse.split()
            for i in range(len(CutString)): #estava range(len(CutString)-1 for some reason antes
                if(CutString[i] =='H'):
                    #Function Amplitude
                    data['data'][0]['y']=highpass(data['data'][0]['y'],int(CutString[i+1])) #aplicar o filtro aos dados y

                elif(CutString[i] =='L'):
                    #Function Low Pass
                    data['data'][0]['y']=lowpass(data['data'][0]['y'], int(CutString[i+1]))

                elif (CutString[i] == 'BP'):
                    #Function Band Pass
                    data['data'][0]['y']=bandpass(data['data'][0]['y'], int(CutString[i+1]), int(CutString[i+2]))

                elif (CutString[i] == 'S'):
                    #Function Smooth
                    smooth_signal = smooth(np.array(data['data'][0]['y'])), int(CutString[i+1]) #estranho porque o smooth retorna uma lista com os dados na 1ªpos e o valor do smooth na segunda
                    data['data'][0]['y']=smooth_signal[0]

                elif (CutString[i] == 'ABS'):
                    #Function Absolute
                    data['data'][0]['y']=np.absolute(data['data'][0]['y'])




        lastclick=n_clicks
        return data

@app.callback(
    dash.dependencies.Output('PreProcessing', 'value'),[
    dash.dependencies.Input('highpass', 'n_clicks'),
    dash.dependencies.Input('lowpass', 'n_clicks'),
    dash.dependencies.Input('bandpass', 'n_clicks'),
    dash.dependencies.Input('smooth', 'n_clicks'),
    dash.dependencies.Input('absolute', 'n_clicks')],
    [dash.dependencies.State('PreProcessing', 'value')]
    # prev_inputs=[
    #         dash.dependencies.PrevInput('button-1', 'n_clicks'),
    #         dash.dependencies.PrevInput('button-2', 'n_clicks'),
    #         dash.dependencies.PrevInput('button-3', 'n_clicks'),
    #         ]
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
    return str(finalStr)


@app.callback(
    dash.dependencies.Output('timevar_graph', 'figure'),
    [dash.dependencies.Input('Radio', 'value'),
     dash.dependencies.Input('slider','value')])
def update_figure(selected_option, value):
    # value=int(value)

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
                name=str(selected_option)
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

    traces.append(go.Scatter(
        x=[time_var['ttv'][value]],
        y=[time_var[str(selected_option)][value]],
        mode='markers'))


        # print(len(time_var['jerk']))
        # print(len(time_var['xt']))
        # filtered_df = dM[selected_option]
    return {
        'data':traces,
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
        if(n_clicks==None):
            traces = []
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

            return {
                'data': traces,
                'layout': go.Layout(
                    showlegend=True,
                    autosize=True,
                    xaxis={'title': 'X position'},
                    yaxis={'title': 'Y position'},
                    legend={'x': 'X', 'y': 'Y'},
                    hovermode='closest'
                )
            }

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
                name="Interpolated Position"
            ))

            return {
                'data': traces,
                'layout': go.Layout(
                    showlegend=True,
                    autosize=True,
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
            x=x,
            y=y,
            mode='markers',
            name='Point Position',
            marker=dict(color='rgb(102,0,0)', size=5, opacity=0.4)
        )
        trace2 = go.Histogram2dcontour(
            x=x,
            y=y,
            name='Position Density',
            ncontours=20,
            colorscale='Hot',
            reversescale=True,
            showscale=False
        )

        data = [trace1, trace2]

        layout = go.Layout(
            showlegend=True,
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

