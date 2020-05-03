import numpy as np


def euclidean_distance(a, b):
    return np.linalg.norm(a-b)

def sigmoid(z):
    return  1 / (1 + np.exp(-z))
