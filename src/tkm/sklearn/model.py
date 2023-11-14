from functools import partial

import jax.numpy as jnp
from jax import random
from jax.lax import fori_loop
import numpy as np
# from jax.numpy.linalg import solve
# from jax.scipy.linalg import solve      # TODO check difference between the two
# from jax.scipy.linalg import kharti_rao

from tkm.features import polynomial, fourier
from tkm.kron import get_dotkron
from jax import jit,vmap

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import (
    check_random_state,
    check_is_fitted,
    check_array,
    check_X_y,
)
from numbers import Real
from typing import Any, List, Optional, Text, Type, Union, Dict, Sequence


from ..model import TensorizedKernelMachine

class TKRC(TensorizedKernelMachine, ClassifierMixin):
    _parameter_contraints: Dict[str, Dict[str, Any]] = {
        "M": [Interval(Real, 1, None, closed="left")],
        "W_init": [Interval(Real, None, None, closed="neither")],
        "l": [Interval(Real, 0, None, closed="left")],
        "numberSweeps": [Interval(Real, 1, None, closed="left")],
        "lengthscale": [Interval(Real, 0, None, closed="left")],
        "R": [Interval(Real, 1, None, closed="left")],
        "key": ["random state"],
        "batch_size": [Interval(Real, 1, None, closed="left")],
        # "class_weight": [StrOptions({"balanced"}), dict, None],
    }
    def __init__(
        self, 
        features='poly',
        M: int = 8,
        R: int = 10,
        lengthscale: float = 0.5,
        l: float = 1e-5,
        numberSweeps: int = 10,
        key=random.PRNGKey(0),
        W_init=None,
        batch_size=None,
        # dotkron = get_dotkron(),
        **kwargs,
    ):
        """
        Args:
            features: kernel function for transforming input data
            M: M_hat number of basis funcitons in one of the CP legs 
            R: rank
            lenghtscale: hyperparameter of fourier features
            get_dotkron: function used for rowwise kronnecker product
            batch_size: if is not None batched_dotkron will be used
            l: peanalty term of regularisation (denoted by lambda)
            numberSweeps: number of ALS sweeps, 1 -> D -> 1 (not including last one)
                1 and D are covered once
                middle is covered twice
                alternative is doing linear pas 1,...,D (not in this code)
            W: device array of weight, used to bypass random init
        """

        super().__init__(features, M, R, lengthscale, l, numberSweeps, key, W_init, batch_size, **kwargs)
            
    def fit(self, x, y, **kwargs):
        super()._jit_funcs(**kwargs)
        check_array(x)
        self.classes_, y = np.unique(y, return_inverse=True)
        x = jnp.asarray(x)
        y = jnp.asarray(y)
        W = self._fit_w(x, y)
        self.W_ = np.array(W)
        return self
    
    def decision_function(self, x, **kwargs):
        check_is_fitted(self, ["W_"])
        check_array(x)
        x = jnp.asarray(x)
        return np.array(super().decision_function(x))
    
    def predict(self, x, **kwargs):
        return np.sign(self.decision_function(x))
    

class TKRR(TensorizedKernelMachine, RegressorMixin):
    def __init__(
        self, 
        features=polynomial,
        M: int = 8,
        R: int = 10,
        lengthscale: float = 0.5,
        l: float = 1e-5,
        numberSweeps: int = 10,
        key=random.PRNGKey(0),
        W_init=None,
        batch_size=None,
        # dotkron = get_dotkron(),
        **kwargs):
        """
        Args:
            features: kernel function for transforming input data
            M: M_hat number of basis funcitons in one of the CP legs 
            R: rank
            lenghtscale: hyperparameter of fourier features
            get_dotkron: function used for rowwise kronnecker product
            batch_size: if is not None batched_dotkron will be used
            l: peanalty term of regularisation (denoted by lambda)
            numberSweeps: number of ALS sweeps, 1 -> D -> 1 (not including last one)
                1 and D are covered once
                middle is covered twice
                alternative is doing linear pas 1,...,D (not in this code)
            W: device array of weight, used to bypass random init
        """
        super().__init__(features, M, R, lengthscale, l, numberSweeps, key, W_init, batch_size, **kwargs)
            
    def fit(self, x, y):
        
        x = jnp.asarray(x)
        y = jnp.asarray(y)
        W = self._fit_w(x, y)
        self.W_ = np.array(W)
        return self
    
    # def decision_function(self, x):
    #     x = jnp.asarray(x)
    #     return np.array(super().predict(x))
    
    def predict(self, x):
        check_is_fitted(self, "W_")
        return self.decision_function(x)
    
    
        
        