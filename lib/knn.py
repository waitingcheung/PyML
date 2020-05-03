import numpy as np
from scipy.stats import mode

from lib.math import euclidean_distance


class KNeighborsClassifier:
    def __init__(self, k=1):
        self.k = k

    def predict(self, X_train, y_train, X_test):
        y_pred = []

        for x in X_test:
            neighbors = self._get_neighbors(X_train, x)
            classes = y_train[neighbors]
            y_pred.append(mode(classes)[0][0])

        return y_pred

    def _get_neighbors(self, X_train, test_instance):
        distances = [(i, euclidean_distance(X_train[i], test_instance)) for i in range(len(X_train))]
        distances.sort(key=lambda tuple: tuple[1])
        neighbors = [distances[i][0] for i in range(self.k)]
        return neighbors
