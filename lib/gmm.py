import numpy as np
from scipy.stats import multivariate_normal

from lib.kmeans import KMeans


class GMM:
    def __init__(self, n_components=1, tol=0.001, max_iter=100, init_params='kmeans', random_state=None):
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter
        self.init_params = init_params
        self.random_state = random_state  

    def predict(self, X):
        weights = self.predict_proba(X)
        return np.argmax(weights, axis=1)

    def predict_proba(self, X):
        likelihood = np.zeros((len(X), self.n_components))
        for i in range(self.n_components):
            distribution = multivariate_normal(self.means_[i], self.covariances_[i])
            likelihood[:, i] = distribution.pdf(X)

        self.proba = likelihood * self.phi
        return self.proba / np.sum(self.proba, axis=1)[:, np.newaxis]

    def _e_step(self, X):
        '''
        Update weights and class probabilities while keeping means and covariances constant.
        '''
        self.weights_ = self.predict_proba(X)
        self.phi = np.mean(self.weights_, axis=0)
        return np.sum(np.log(np.sum(self.proba, axis=1)))

    def _m_step(self, X):
        '''
        Update means and covariances while keeping weights constant.
        '''
        for i in range(self.n_components):
            weight = self.weights_[:, [i]]
            total_weight = np.sum(weight)
            self.means_[i] = np.sum(weight * X, axis=0) / total_weight
            self.covariances_[i] = np.cov(X.T, aweights=(weight / total_weight).flatten(), bias=True)

    def fit(self, X):
        np.random.seed(seed=self.random_state)

        weights_shape = X.shape[0], self.n_components

        if self.init_params == 'kmeans':
            kmeans = KMeans(self.n_components, random_state=self.random_state)
            kmeans.fit(X)
            classes = kmeans.predict(X)
            self.weights_ = np.zeros(weights_shape)
            self.weights_[np.arange(X.shape[0]), classes] = 1
        else:
            self.weights_ = np.random.rand(*weights_shape)
        
        self.phi = np.random.rand(*weights_shape)

        random_row = np.random.randint(0, len(X), self.n_components)
        self.means_ = [X[r, :] for r in random_row]
        self.covariances_ = [np.cov(X.T) for _ in range(self.n_components)]

        best_likelihood = 0
        for _ in range(self.max_iter):
            log_likelihood = self._e_step(X)
            self._m_step(X)

            if abs(best_likelihood - log_likelihood) < self.tol:
                break

            best_likelihood = log_likelihood
