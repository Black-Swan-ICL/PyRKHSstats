# TODO use OOP approach ? Or leave as just a module ?
import copy

import numpy as np
import pandas as pd


class ArraysNonConformable(Exception):

    pass


# TODO document
# TODO test
def to_numpy_array(x):
    x = copy.deepcopy(x)
    if isinstance(x, np.ndarray):
        res = x
    elif isinstance(x, list):
        res = np.asarray(x)
    # TODO is that not redundant ?
    elif isinstance(x, pd.core.series.Series) or isinstance(x, pd.Series):
        res = x.values
    elif isinstance(x, pd.DataFrame):
        res = x.values
    else:
        msg = "x must a list, numpy array, pandas series or pandas dataframe !"
        raise TypeError(msg)

    return res


# TODO document
def check_conformable(*args):

    shapes = [arg.shape for arg in args]

    if len(set(shapes)) != 1:
        raise ArraysNonConformable

    return True
