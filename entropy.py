import numpy as np


def deviation(y):
    return y.std()


def gini(y):
    # compute gini impurity
    m = y.sum(axis=0)
    size = y.shape[0]
    e = (m / size) ** 2
    return 1.0 - np.sum(e)


def infgain(y):
    m - y.sum(axis=0)
    size = y.shape[0]
    e = [p * np.log2(p / size) / size for p in m if p != 0.0]
    return -np.sum(e)
