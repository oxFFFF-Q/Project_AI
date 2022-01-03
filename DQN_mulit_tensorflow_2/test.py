import pandas as pd
import pickle
import numpy as np
from keras.utils import to_categorical
ss = np.load("pre_trained/FFA-Dataset-actions.npy", allow_pickle=True)
ss=to_categorical(ss,6)

print(ss)
