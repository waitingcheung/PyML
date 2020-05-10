# PyML

A minimalistic implementation of common machine learning algorithms.

## Overview

[**Linear Regression**](#linear-regression) | [**Logistic Regression**](#logistic-regression) | [**SVM**](#svm)
:---: | :---: | :---:
![](images/linear_regression.png) | ![](images/logistic_regression.png) | ![](images/svm.png) 
[**Naive Bayes**](#naive-bayes) | [**CART**](#cart) | [**k-NN**](#k-NN)
 ![](images/naive_bayes.png) | ![](images/cart.png) | ![](images/knn.png) 

## Usage

### Linear Regression

[Full Example](examples/linear_regression.ipynb)

```python
import numpy as np

from lib.linear_regression import LinearRegression

X = 30 * np.random.random((50, 1))
y = 0.5 * X + 1.0 + np.random.normal(size=X.shape)

model = LinearRegression(verbose=True)
model.fit(X, y)

X_new = np.linspace(0, 30, 100)
y_new = model.predict(X_new[:, np.newaxis])
```

### Logistic Regression

[Full Example](examples/logistic_regression.ipynb)

```python
from sklearn.datasets import make_blobs

from lib.logistic_regression import LogisticRegression

X, y = make_blobs(n_samples=80, centers=2, random_state=0)

model = LogisticRegression(verbose=True)
model.fit(X, y)

y_pred = model.predict(X)
```

### SVM

Note: This implements [PEGASOS: Primal Estimated sub-GrAdient SOlver for SVM](https://ttic.uchicago.edu/~nati/Publications/PegasosMPB.pdf) instead of the standard SVM.

#### Linear Separable

[Full Example](examples/svm_linear.ipynb)

```python
from sklearn.datasets import make_blobs

from lib.pegasos import PegasosSVM

X, y = make_blobs(n_samples=40, centers=2, random_state=6)
y = np.where(y > 0, 1, -1)

model = PegasosSVM(kernel='linear', max_iter=100)
model.fit(X, y)

y_pred = model.predict(X)
```

#### Non-linear Separable

[Full Example](examples/svm_nonlinear.ipynb)

```python
from sklearn.datasets import make_circles

from lib.pegasos import PegasosSVM

X, y = make_circles(noise=0.2, factor=0.5, random_state=1)
y = np.where(y > 0, 1, -1)

model = PegasosSVM(kernel='rbf', max_iter=100)
model.fit(X, y)

y_pred = model.predict(X)
```

### Naive Bayes

[Full Example](examples/naive_bayes.ipynb)

```python
from sklearn.datasets import make_blobs

from lib.naive_bayes import NaiveBayes

X, y = make_blobs(n_samples=60, centers=2, random_state=2, cluster_std=1.5)

model = NaiveBayes()
model.fit(X, y)

y_pred = model.predict(X)
```

### CART
[Full Example](examples/cart.ipynb)

```python
from sklearn.datasets import load_iris

from lib.cart import DecisionTreeClassifier

X, y = load_iris(return_X_y=True)

model = DecisionTreeClassifier()
model.fit(X, y)
y_pred = model.predict(X)
```

### k-NN
[Full Example](examples/knn.ipynb)

```python
from sklearn.datasets import make_moons

from lib.knn import KNeighborsClassifier

X, y = make_moons(noise=0.3, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = KNeighborsClassifier(k=3)
y_pred = model.predict(X_train, y_train, X_test)
```

## References

- [fast.ai Wiki](http://wiki.fast.ai/index.php/Main_Page)
- [Implementing Naive Bayes in 2 minutes with Python](https://towardsdatascience.com/implementing-naive-bayes-in-2-minutes-with-python-3ecd788803fe)
- [Decision Tree from Scratch in Python](https://towardsdatascience.com/decision-tree-from-scratch-in-python-46e99dfea775)
- [Develop k-Nearest Neighbors in Python From Scratch](https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/)
- [Implementing PEGASOS: Primal Estimated sub-GrAdient SOlver for SVM, Logistic Regression and Application in Sentiment Classification (in Python)](https://sandipanweb.wordpress.com/2018/04/29/implementing-pegasos-primal-estimated-sub-gradient-solver-for-svm-using-it-for-sentiment-classification-and-switching-to-logistic-regression-objective-by-changing-the-loss-function-in-python/)