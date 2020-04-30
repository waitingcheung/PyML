import sys

import numpy as np


class LogisticRegression:
    def __init__(self, weights=None, lr=1e-4, verbose=False):
        self.weights = weights
        self.lr = lr
        self.verbose = verbose

    def sigmoid(self, z):
        return  1 / (1 + np.exp(-z))

    def predict_proba(self, x):
        '''
        Returns 1D array of probabilities
        that the class label == 1
        '''
        if self.weights is None:
            self.weights = np.random.rand(x.shape[-1])

        return self.sigmoid(np.dot(x, self.weights))

    def log_loss(self, x, y):
        '''
        Features:(100,3)
        Labels: (100,1)
        Weights:(3,1)
        Returns 1D matrix of predictions
        Cost = ( log(predictions) + (1-labels)*log(1-predictions) ) / len(labels)
        '''
        predictions = self.predict_proba(x)
        
        #Take the error when label=1
        class1_loss = -y*np.log(predictions)

        #Take the error when label=0
        class2_loss = (1-y)*np.log(1-predictions)
        
        #Take the sum of both costs
        loss = class1_loss - class2_loss

        return np.mean(loss)

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
            loss = self.log_loss(x, y)

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
