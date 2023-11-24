import sklearn
from sklearn.model_selection import cross_validate, GridSearchCV
import tkm
from tkm.model import TensorizedKernelMachine as TKM
from tkm.model import TensorizedKernelClassifier as TKC
from tensorlibrary.learning.t_krr import CPKRR
import jax.numpy as jnp

from sklearn import datasets
# X, y = datasets.load_breast_cancer(return_X_y=True)
# # convert y to -1,1
# y = 2*y - 1
# N = len(y)
#
# # %% tensorlibrary method
import pickle
# import numpy as np
with open('../test/test_model.pkl', 'rb') as f:
    model_params = pickle.load(f)
f.close()
X = model_params['X']
y = model_params['y']
W = model_params['W']
w_init = model_params['W_init']
features = model_params['feature_map']
l = model_params['l']
M = model_params['M']
num_sweeps = model_params['num_sweeps']
R = model_params['R']

# w_init_s = [np.array(model_params['W_init'][i]) for i in range(len(model_params['W_init']))]
# model = CPKRR(feature_map='poly', max_rank=R, reg_par=l/N, num_sweeps=num_sweeps, M=5)#, w_init=w_init_s)
# model = model.fit(np.array(X), np.array(y))
# print(model.score(np.array(X), np.array(y)))
# w = model.weights_

# %% pickle
# import pickle
# model_params = {}
# model_params['W'] = jnp.array(model.weights_)
# model_params['feature_map'] = model.feature_map
# model_params['l'] = model.reg_par*N
# model_params['M'] = model.M
# model_params['num_sweeps'] = model.num_sweeps
# model_params['R'] = model.max_rank
# model_params['X'] = jnp.array(X)
# model_params['y'] = jnp.array(y)
# model_params['W_init'] = jnp.array(model.w_init)
# # with open('../test/test_model.pkl', 'wb') as f:
# #     pickle.dump(model_params, f)
#
# # %% check with sklearn
# grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
# grid.fit(X, y)
# print(grid.best_params_)
# JAX_CHECK_TRACER_LEAKS=True
import os
os.environ['JAX_CHECK_TRACER_LEAKS']='True'

# %% check with TKM
#
# X = jnp.array(X)
# y = jnp.array(y)
# param_grid = {'M': [2]}
# model = TKC(features='poly', R=2, l=1e-10, num_sweeps=10, M=5)
# # grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
# model.fit(X, y)
# print(model.score(X,y))
# # print(grid.best_params_)
#%%
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X = jnp.array(X)
#%%
import numpy as np
model = TKC(features=features, R=R, l=l, num_sweeps=num_sweeps, M=4)#, W_init=w_init)
model = model.fit(X, y)
y_hat = model.predict(X)
print(model.score(X,y))
accuracy = accuracy_score(np.array(y), np.array(y_hat))
wj = [model.W_[i] for i in range(len(model.W_))]
print(accuracy)
# assert jnp.allclose(model.W_, W)

#%% perform a grid search
param_grid = {'M': [2, 3, 4, 5]}
grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=1)
grid.fit(X, y)
print(grid.best_params_)

# %% check with sklearn
model = model.set_params(M=5)
model = model.fit(X, y)