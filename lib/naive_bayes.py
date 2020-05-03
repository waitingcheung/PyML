import operator

import numpy as np
from scipy.stats import norm


class NaiveBayes:        
    def predict(self, X):
        y_pred = []

        for x in X:
            probas_x = {}

            # P(X1, X2, ... Xn | C) = P(C|X1) * P(C|X2) * ... * P(C|Xn) * P(C)
            for c in self.labels:
                probas_x[c] = self.probas[c]
                
                for feature in x:
                    probas_x[c] *= norm.pdf(feature, self.means[c], self.stds[c])

            y_pred.append(max(probas_x.items(), key=operator.itemgetter(1))[0])

        return y_pred

    def fit(self, X, y):
        self.labels = np.unique(y)
        self.means = [X[y == c].mean() for c in self.labels]
        self.stds = [X[y == c].std() for c in self.labels]
        self.probas = np.asarray([len(X[y == c]) for c in self.labels]) / X.shape[0]
