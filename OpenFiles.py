import pandas as pd
import pickle
import seaborn as sns
import numpy as np
import easygui

path = easygui.fileopenbox()

with open(path, 'rb') as handle:
    collection= pickle.load(handle)

