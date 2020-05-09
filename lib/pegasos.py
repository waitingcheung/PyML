import numpy as np


class PegasosSVM:
    def __init__(self, kernel='rbf', degree=3, gamma=1, coef0=0.0, lambda_=1, max_iter=10):
        if kernel and kernel not in ['linear', 'poly', 'rbf', 'sigmoid']:
            raise ValueError('Kernel must be one of \'linear\', \'poly\', \'rbf\', \'sigmoid\'')

        self.kernel_map = {
            'linear': self._linear,
            'poly': self._poly,
            'rbf': self._rbf,
            'sigmoid': self._sigmoid
        }
        self.kernel = self.kernel_map[kernel] if kernel else None
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.lambda_ = lambda_
        self.max_iter = max_iter

    def _linear(self, x, y):
        return np.dot(x, y)

    def _poly(self, x, y):
        return (self.gamma * np.dot(x, y) + self.coef0) ** self.degree

    def _rbf(self, x, y):
        return np.exp(-self.gamma * np.linalg.norm(x - y) ** 2)

    def _sigmoid(self, x, y):
        return np.tanh(self.gamma * np.dot(x, y) + self.coef0)

    def predict(self, X):
        if self.kernel:
            m = X.shape[0]
            scores = np.zeros(m)

            for i in range(m):
                score = 0
                for k in range(self.weights.shape[0]):
                    score += self.weights[k] * self.kernel(X[i], self.X_train[k]) * self.y_train[k]
                scores[i] = score
        else:
            scores = np.dot(X, self.weights)

        return np.where(scores > 0, 1, -1)

    def fit(self, X, y):
        m, n = X.shape[0], X.shape[1]
        self.weights = np.zeros(m if self.kernel else n) 

        for i in range(self.max_iter):
            eta = 1. / (self.lambda_ * (i + 1))            
            idx = np.random.randint(0, m)
            X_i, y_i = X[idx], y[idx]

            if self.kernel:
                score = 0
                for k in range(m):
                    score += self.weights[k] * self.kernel(X_i, X[k]) * y[k]

                self.weights[idx] *= (1 - eta * self.lambda_)
                self.weights[idx] += 1 if y_i * score < 1 else 0

                self.X_train = X
                self.y_train = y
            else:
                score = np.dot(self.weights, X_i)
                self.weights *= (1 - eta * self.lambda_)
                self.weights += eta * y_i * X_i if y_i * score < 1 else 0
