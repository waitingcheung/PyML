import sys

import numpy as np

from lib.loss import cross_entropy_loss
from lib.math import sigmoid


class LogisticRegression:
    def __init__(self, weights=None, lr=1e-3, tol=1e-4, verbose=False):
        self.weights = weights
        self.lr = lr
        self.tol = tol
        self.verbose = verbose

    def predict_proba(self, X):
        '''
        Returns 1D array of probabilities
        that the class label == 1
        '''
        if self.weights is None:
            self.weights = np.random.rand(X.shape[-1])

        return sigmoid(np.dot(X, self.weights))

    # Vectorized Gradient Descent
    # gradient = X.T * (X*W - y) / N
    # gradient = features.T * (predictions - labels) / N
    def _update_weights(self, X, y):   
        predictions = self.predict_proba(X)
        gradient = np.dot(X.T,  predictions - y)
        self.weights -= self.lr * (gradient / len(X))

    def _decision_boundary(self, prob):
        return 1 if prob >= .5 else 0

    def predict(self, X):
        '''
        preds = N element array of predictions between 0 and 1
        returns N element array of 0s (False) and 1s (True)
        '''
        proba = self.predict_proba(X)
        decision_boundary = np.vectorize(self._decision_boundary)  #vectorized function
        return decision_boundary(proba).flatten()

    def fit(self, X, y):
        if self.weights is None:
            self.weights = np.random.rand(X.shape[-1])

        best_loss, best_itr = None, 0

        for i in range(sys.maxsize):
            self._update_weights(X, y)
            y_pred = self.predict_proba(X)
            loss = cross_entropy_loss(y, y_pred)

            if self.verbose and i % int(1 / self.lr) == 0:
                print(f'iter {i :>7} \t loss {loss :>7.3f}')

            if best_loss and abs(loss - best_loss) < self.tol and i - best_itr >= 2:
                break

            if best_loss is None or loss < best_loss:
                best_loss, best_itr = loss, i
