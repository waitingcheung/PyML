import operator

import numpy as np
from scipy.stats import norm


class NaiveBayes:
    def __init__(self):
        self.pdf = np.vectorize(norm.pdf)

    def predict(self, X):
        y_pred = []

        for x in X:
            best_proba, best_label = 0, None

            for c in self.labels:
                # P(X1, X2, ... Xn | C) = P(C|X1) * P(C|X2) * ... * P(C|Xn) * P(C)
                proba = np.prod(self.pdf(x, self.means[c], self.stds[c])) * self.probas[c]
                if proba > best_proba:
                    best_proba, best_label = proba, c

            y_pred.append(best_label)

        return y_pred

    def fit(self, X, y):
        self.labels = np.unique(y)
        self.means = [X[y == c].mean() for c in self.labels]
        self.stds = [X[y == c].std() for c in self.labels]
        self.probas = np.asarray([len(X[y == c]) for c in self.labels]) / X.shape[0]
