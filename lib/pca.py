import numpy as np


class PCA:
    def __init__(self, n_components=None, whiten=False):
        self.n_components = n_components
        self.whiten = whiten

    def fit(self, X):
        if self.n_components is None:
            self.n_components = min(X.shape[0], X.shape[1]) - 1

        X = X.astype(np.float64)

        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_

        if self.whiten:
            self.std = np.std(X, axis=0)
            X /= self.std

        covariance_matrix = np.dot(X.T, X) / X.shape[0]
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        indices = np.argsort(eigenvalues)[::-1][:self.n_components]
        self.eigenvalues, self.eigenvectors = eigenvalues[indices], eigenvectors[:,indices]

    def transform(self, X):
        X = X.astype(np.float64)

        X -= self.mean_

        if self.whiten:
            X /= self.std

        return np.dot(X, self.eigenvectors)

    @property
    def explained_variance_ratio_(self):
        return self.eigenvalues / np.sum(self.eigenvalues)
