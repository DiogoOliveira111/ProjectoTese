from OpenFiles import collection
import pandas as pd
from WBMTools.sandbox.interpolation import interpolate_data
from tkinter import Tk, Label
import GrammarofTime.gotstools as gt
import matplotlib.pyplot as plt
import numpy as np
from Load import load

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
        my_text = Label(root, text='The Collection chosen has no Mouse Data ')
        my_text.pack()
        root.mainloop()
        exit()


MouseDict = dict(t=MouseTime, x=MouseX, y=MouseY)
dM = pd.DataFrame.from_dict(MouseDict)


# time_var,space_var=interpolate_data(dM,t_abandon=20)
# v = time_var["vt"]
# vx = time_var["vx"]


str = "xt a"
s = load(dM, str)
# print(s)

#s = np.array([v, vx])
#print(time_var['vt'])

#plt.plot(time_var['vt'])


cfg={"pre_processing":"",
     "connotation": "⇞ 0.8",
     "expression": "1+"}

matches=gt.ssts(s, cfg)
print(matches)

gt.plot_matches(s, matches, scatter=True)

plt.show()