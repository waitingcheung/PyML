import sys

import numpy as np

from lib.loss import mse_loss


class LinearRegression:
    def __init__(self, weight=1, bias=2, lr=1e-3, tol=1e-4, verbose=False):
        self.weight = weight
        self.bias = bias
        self.lr = lr
        self.tol = tol
        self.verbose = verbose

    def predict(self, X):
        return self.weight * X + self.bias

    def _update_weights(self, X, y):
        '''
        Calculate partial derivatives
        MSE = (y - (mx + b))^2
        dM = -2x(y - (mx + b))
        dB = -2(y - (mx + b))
        '''
        y_pred = self.predict(X)
        weight_deriv = -2 * X * (y - y_pred)
        bias_deriv = -2 * (y - y_pred)

        self.weight -= self.lr * np.mean(weight_deriv)
        self.bias -= self.lr * np.mean(bias_deriv)

    def fit(self, X, y):
        best_loss, best_itr = None, 0

        for i in range(sys.maxsize):
            self._update_weights(X, y)
            y_pred = self.predict(X)
            loss = mse_loss(y, y_pred)

            if self.verbose and i % int(1 / self.lr) == 0:
                print(f'iter {i :>7} \t weight {self.weight :>7.3f} \t bias {self.bias :>7.3f} \t loss {loss :>7.3f}')

            if best_loss and abs(loss - best_loss) < self.tol and i - best_itr >= 2:
                break

            if best_loss is None or loss < best_loss:
                best_loss, best_itr = loss, i
