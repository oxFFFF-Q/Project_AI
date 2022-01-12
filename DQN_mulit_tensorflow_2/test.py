import random

import pandas as pd
import pickle
import numpy as np
from keras.utils import to_categorical

grids = [4, 4, 4, 4]
if 4 in grids and all((grid in [1, 2, 3, 4]) for grid in grids):
    print("ok")
