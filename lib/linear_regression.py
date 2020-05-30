import sys

import numpy as np

from lib.loss import mse_loss


class LinearRegression:
    def __init__(self, lr=1e-3, tol=1e-4, verbose=False):
        self.lr = lr
        self.tol = tol
        self.verbose = verbose

    def predict(self, X):
        if X.shape[1] < len(self.weights):
            bias = np.ones(shape=(X.shape[0], 1))
            X = np.append(X, bias, axis=1)
        return np.dot(X, self.weights)

    def _update_weights(self, X, y):
        '''
        Calculate partial derivatives
        MSE = (y - (w1x1 + w2x2 + ... + wnxn))^2
        d_wk = -x(y - wkxk)
        '''
        y_pred = self.predict(X)
        error = y - y_pred
        gradient = np.dot(-X.T, error) / X.shape[0]
        self.weights -= self.lr * gradient

    def fit(self, X, y, weights=None):
        bias = np.ones(shape=(X.shape[0], 1))
        X = np.append(X, bias, axis=1)
        y = np.squeeze(y)
        self.weights = weights if weights else np.zeros(X.shape[1])

        best_loss, best_itr = None, 0

        for i in range(sys.maxsize):
            self._update_weights(X, y)
            y_pred = self.predict(X)
            loss = mse_loss(y, y_pred) / 2

            if self.verbose and i % int(1 / self.lr) == 0:
                print(f'iter {i :>7} \t loss {loss :>7.3f}')

            if best_loss and abs(loss - best_loss) < self.tol and i - best_itr >= 2:
                break

            if best_loss is None or loss < best_loss:
                best_loss, best_itr = loss, i
