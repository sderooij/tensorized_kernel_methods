import sklearn
from sklearn.model_selection import cross_validate, GridSearchCV
import tkm
from tkm.model import TensorizedKernelMachine as TKM
from tkm.model import TensorizedKernelClassifier as TKC
from tensorlibrary.learning.t_krr import CPKRR
import jax.numpy as jnp

from sklearn import datasets
X, y = datasets.load_breast_cancer(return_X_y=True)
# convert y to -1,1
y = 2*y - 1
N = len(y)

# %% tensorlibrary method
param_grid = {'M': [2, 5]}
model = CPKRR(feature_map='poly', max_rank=2, reg_par=1e-10/N, num_sweeps=10)
grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid.fit(X, y)
print(grid.best_params_)

# %% check with TKM

X = jnp.array(X)
y = jnp.array(y)
param_grid = {'M': [2]}
model = TKC(features='poly', R=2, l=1e-10, num_sweeps=10)
grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid.fit(X, y)
print(grid.best_params_)