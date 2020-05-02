import sys

import numpy as np

from lib.loss import cross_entropy_loss
from lib.math import sigmoid


class LogisticRegression:
    def __init__(self, weights=None, lr=1e-4, verbose=False):
        self.weights = weights
        self.lr = lr
        self.verbose = verbose

    def predict_proba(self, x):
        '''
        Returns 1D array of probabilities
        that the class label == 1
        '''
        if self.weights is None:
            self.weights = np.random.rand(x.shape[-1])

        return sigmoid(np.dot(x, self.weights))

    # Vectorized Gradient Descent
    # gradient = X.T * (X*W - y) / N
    # gradient = features.T * (predictions - labels) / N
    def update_weights(self, x, y):
        '''
        Features:(200, 3)
        Labels: (200, 1)
        Weights:(3, 1)
        '''    
        predictions = self.predict_proba(x)
        gradient = np.dot(x.T,  predictions - y)
        self.weights -= self.lr * (gradient / len(x))

    def decision_boundary(self, prob):
        return 1 if prob >= .5 else 0

    def predict(self, x):
        '''
        preds = N element array of predictions between 0 and 1
        returns N element array of 0s (False) and 1s (True)
        '''
        proba = self.predict_proba(x)
        decision_boundary = np.vectorize(self.decision_boundary)  #vectorized function
        return decision_boundary(proba).flatten()

    def fit(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)

        if self.weights is None:
            self.weights = np.random.rand(x.shape[-1])

        count = 0
        prev_loss = 0
        epsilon = 1e-8
        for i in range(sys.maxsize):
            self.update_weights(x, y)
            y_pred = self.predict_proba(x)
            loss = cross_entropy_loss(y, y_pred)

            if self.verbose and i % int(1 / self.lr) == 0:
                print(f'iter {i :>7} \t loss {loss :>7.3f}')

            if abs(loss - prev_loss) < epsilon:
                count += 1
                if count == 2: break
            else:
                count = 0
            
            prev_loss = loss

    def accuracy(self, predicted_labels, actual_labels):
        diff = predicted_labels - actual_labels
        return 1.0 - (float(np.count_nonzero(diff)) / len(diff))
