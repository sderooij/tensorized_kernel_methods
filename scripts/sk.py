import sklearn
from sklearn.model_selection import cross_validate, GridSearchCV
import tkm
from tkm.model import TensorizedKernelMachine as TKM
import jax.numpy as jnp

from sklearn import datasets
X, y = datasets.load_breast_cancer(return_X_y=True)
# convert y to -1,1
y = 2*y - 1

X = jnp.array(X)
y = jnp.array(y)
param_grid = {'M': [2, 3]}
model = TKM(features='poly', R=2, l=1e-9, numberSweeps=10)
grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid.fit(X, y)
print(grid.best_params_)