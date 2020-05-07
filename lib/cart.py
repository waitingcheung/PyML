import numpy as np


class Node:
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None

class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.n_classes_ = len(set(y))
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y)

    def predict(self, X):
        return [self._predict(inputs) for inputs in X]

    def _predict(self, inputs):
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class

    def _gini_impurity(self, class_count, n_classes):
        return 1 - np.sum((class_count / n_classes) ** 2)

    def _best_split(self, X, y):
        m = y.size
        
        class_count = np.bincount(y, minlength=self.n_classes_)
        best_gini = self._gini_impurity(class_count, m)
        best_idx, best_thr = None, None

        for feature_idx in range(self.n_features_):
            thresholds, classes = zip(*sorted(zip(X[:, feature_idx], y)))
            
            num_left = np.zeros(self.n_classes_)
            num_right = class_count.copy()

            # Try all possible splits
            for i in range(1, m):
                # Don't split features of identical values
                if thresholds[i] == thresholds[i - 1]: continue

                # Update the number of samples per class on both sides
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1

                gini_left = self._gini_impurity(num_left, i)
                gini_right = self._gini_impurity(num_right, m - i)
                gini = (i * gini_left + (m - i) * gini_right) / m
                
                if gini < best_gini:
                    best_gini = gini
                    best_idx = feature_idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2

        return best_idx, best_thr

    def _grow_tree(self, X, y, depth=0):
        class_count = np.bincount(y)
        predicted_class = np.argmax(class_count)
        node = Node(predicted_class=predicted_class)
        
        if len(set(y)) <= 1 or len(class_count) == 1: return node

        if self.max_depth is None or (self.max_depth and depth < self.max_depth):
            idx, thr = self._best_split(X, y)

            indices_left = X[:, idx] < thr
            X_left, y_left = X[indices_left], y[indices_left]
            X_right, y_right = X[~indices_left], y[~indices_left]

            node.feature_index = idx
            node.threshold = thr
            node.left = self._grow_tree(X_left, y_left, depth + 1)
            node.right = self._grow_tree(X_right, y_right, depth + 1)

        return node
