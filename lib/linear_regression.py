import sys
import numpy as np

class LinearRegression:
    def __init__(self, weight=1, bias=2, lr=1e-4):
        self.weight = weight
        self.bias = bias
        self.lr = lr

    def predict(self, x):
        return self.weight * x + self.bias

    def mse_loss(self, x, y):
        loss = (y - (self.weight * x + self.bias)) ** 2
        return np.mean(loss)

    def update_weights(self, x, y):
        # Calculate partial derivatives
        # -2x(y - (mx + b))
        weight_deriv = -2 * x * (y - (self.weight * x + self.bias))
        # -2(y - (mx + b))
        bias_deriv = -2 * (y - (self.weight * x + self.bias))

        self.weight -= self.lr * np.mean(weight_deriv)
        self.bias -= self.lr * np.mean(bias_deriv)

    def fit(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)

        count = 0
        prev_loss = 0
        epsilon = 1e-8
        for _ in range(sys.maxsize):
            self.update_weights(x, y)
            loss = self.mse_loss(x, y)

            if abs(loss - prev_loss) < epsilon:
                count += 1
                if count == 2: break
            else:
                count = 0
            
            prev_loss = loss
