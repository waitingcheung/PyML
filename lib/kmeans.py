from copy import deepcopy

import numpy as np

from lib.math import euclidean_distance


class KMeans:
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None

    def predict(self, X):
        return np.array([np.argmin(euclidean_distance(x, self.cluster_centers_, axis=1)) for x in X])

    def fit(self, X):
        np.random.seed(seed=self.random_state)

        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        
        cluster_centers_ = np.random.randn(self.n_clusters, X.shape[1]) * std + mean
        cluster_centers_old_ = np.zeros(cluster_centers_.shape)

        distances = np.zeros((X.shape[0], self.n_clusters))

        for _ in range(self.max_iter):
            if euclidean_distance(cluster_centers_, cluster_centers_old_) < self.tol:
                break

            for i in range(self.n_clusters):
                distances[:, i] = euclidean_distance(X, cluster_centers_[i], axis=1)

            clusters = np.argmin(distances, axis=1)

            cluster_centers_old_ = deepcopy(cluster_centers_)

            for i in range(self.n_clusters):
                cluster_centers_[i] = np.mean(X[clusters == i], axis=0)

        self.cluster_centers_ = cluster_centers_        
