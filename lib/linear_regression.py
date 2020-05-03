import sys

import numpy as np

from lib.loss import mse_loss


class LinearRegression:
    def __init__(self, weight=1, bias=2, lr=1e-4, verbose=False):
        self.weight = weight
        self.bias = bias
        self.lr = lr
        self.verbose = verbose

    def predict(self, X):
        return self.weight * X + self.bias

    def _update_weights(self, X, y):
        y_pred = self.predict(X)

        # Calculate partial derivatives
        # -2x(y - (mx + b))
        weight_deriv = -2 * X * (y - y_pred)
        # -2(y - (mx + b))
        bias_deriv = -2 * (y - y_pred)

        self.weight -= self.lr * np.mean(weight_deriv)
        self.bias -= self.lr * np.mean(bias_deriv)

    def fit(self, X, y):
        count = 0
        prev_loss = 0
        epsilon = 1e-8
        for i in range(sys.maxsize):
            self._update_weights(X, y)
            y_pred = self.predict(X)
            loss = mse_loss(y, y_pred)

            if self.verbose and i % int(1 / self.lr) == 0:
                print(f'iter {i :>7} \t weight {self.weight :>7.3f} \t bias {self.bias :>7.3f} \t loss {loss :>7.3f}')

            if abs(loss - prev_loss) < epsilon:
                count += 1
                if count == 2: break
            else:
                count = 0
            
            prev_loss = loss
