from WBMTools.sandbox.interpolation import interpolate_data
#from Processing import dM
import numpy as np


signal=[]


def load(dM, str):
    time_var, space_var = interpolate_data(dM, t_abandon=20)
    command= str.split() #each position in vector is the command ie V Vx Vy

    for i in range(len(command)):

        if (command[i] in ("vt, vx, vy, a, jerk, xt, yt, tt, ttv")):
            print(command[i])

            signal.append(time_var[command[i]])


        elif(command[i] in ("xs, ys, l_strokes, straightness, jitter, s, ss, angles, w, curvatures, var_curvatures")):
            signal.append(space_var[command[i]])

    return signal

