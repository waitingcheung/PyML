# PyML

A minimalistic implementation of common machine learning algorithms.

## Overview

Linear Regression | Logistic Regression
:---: | :---:
![](images/linear_regression.png) | ![](images/logistic_regression.png)

## Usage

### Linear Regression

[Full Example](examples/linear_regression.ipynb)

```python
import numpy as np

from lib.linear_regression import LinearRegression

x = 30 * np.random.random((50, 1))
y = 0.5 * x + 1.0 + np.random.normal(size=x.shape)

model = LinearRegression(verbose=True)
model.fit(x, y)

x_new = np.linspace(0, 30, 100)
y_new = model.predict(x_new[:, np.newaxis])
```

### Logistic Regression

[Full Example](examples/logistic_regression.ipynb)

```python
from sklearn.datasets import make_blobs

from lib.logistic_regression import LogisticRegression

x, y = make_blobs(n_samples=80, centers=2, random_state=0)

model = LogisticRegression(verbose=True)
model.fit(x, y)

y_pred = model.predict(x)
```
