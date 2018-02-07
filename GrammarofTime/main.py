import pandas as pd
import pickle
from WBMTools.sandbox.interpolation import interpolate_data
import GrammarofTime.gotstools as gt
import numpy as np

MouseX = []
MouseY = []
MouseTime = []
with open("collection_1482244484.pkl", 'rb') as handle:
        collection = pickle.load(handle)

for i in collection:
    event = collection[i]

    if (event['Type'] == 'Mouse'):

        data = event['Data'].split(';')
        # print (data[-1])

        if (i == 0):
                initial_time = float(data[-1])
                MouseTime.append(0)
        else:
            MouseTime.append((float(data[-1]) - initial_time) / 1000)
        MouseX.append(float(data[2]))
        MouseY.append(float(data[3]))

MouseDict = dict(t=MouseTime, x=MouseX, y=MouseY)
dM = pd.DataFrame.from_dict(MouseDict)

time_var,space_var=interpolate_data(dM,t_abandon=20)

cfg={ "pre_processing": "V",
      "connotation": "⇞0.8",
      "expression": "1+"}

matches=gt.ssts(dM, cfg)
gt.plot_matches(dM, matches)