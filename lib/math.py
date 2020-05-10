import numpy as np


def euclidean_distance(a, b, axis=None):
    return np.linalg.norm(a - b, axis=axis)

def sigmoid(z):
    return  1 / (1 + np.exp(-z))
