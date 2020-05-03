# PyML

A minimalistic implementation of common machine learning algorithms.

## Overview

Linear Regression | Logistic Regression | Naive Bayes
:---: | :---: | :---:
![](images/linear_regression.png) | ![](images/logistic_regression.png) | ![](images/naive_bayes.png)

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

## References

- [fast.ai Wiki](http://wiki.fast.ai/index.php/Main_Page)
