import pandas as pd
import pickle
import numpy as np
from keras.utils import to_categorical

ss = np.array([[1, 2, 3],
               [4, 5, 6],
               [5, 6, 7]])


print(np.argwhere(ss == 5))
