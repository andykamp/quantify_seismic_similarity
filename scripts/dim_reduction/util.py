from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import pandas as pd

from sklearn.utils import shuffle

def relu(x):
    return x * (x > 0)


def error_rate(p, t):
    return np.mean(p != t)


def init_weights(shape):
    w = np.random.randn(*shape) / np.sqrt(sum(shape))
    return w.astype(np.float32)
