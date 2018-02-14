from WBMTools.sandbox.interpolation import interpolate_data
#from Processing import dM
import numpy as np





def load(dM, str):
    signal = []
    time_var, space_var = interpolate_data(dM, t_abandon=20)
    command = str.split()  # each position in vector is the command ie V Vx Vy
    print(command)
    if(len(command) >1):

        for i in range(len(command)):

            if (command[i] in ("vt, vx, vy, a, jerk, xt, yt, tt, ttv")):

                signal.append(time_var[command[i]])
                print(signal)

            elif(command[i] in ("xs, ys, l_strokes, straightness, jitter, s, ss, angles, w, curvatures, var_curvatures")):
                signal.append(space_var[command[i]])
    else:
        if (str in ("vt, vx, vy, a, jerk, xt, yt, tt, ttv")):

            signal=time_var[str]


        elif (str in ("xs, ys, l_strokes, straightness, jitter, s, ss, angles, w, curvatures, var_curvatures")):
            signal=space_var[str]

    return signal

