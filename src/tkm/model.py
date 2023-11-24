from functools import partial

import jax.numpy as jnp
from jax import random
from jax.lax import fori_loop
# from jax.numpy.linalg import solve
# from jax.scipy.linalg import solve      # TODO check difference between the two
# from jax.scipy.linalg import kharti_rao

from tkm.features import polynomial, fourier
from tkm.kron import get_dotkron
from tkm.metrics import accuracy, rmse
from tkm._cpkrr import _init_g, _init_reg
from jax import jit, vmap

from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, ClassifierMixin

# from jax.config import config     # uncomment for debugging
# config.update('jax_disable_jit', True)


class BaseTKM(ABC):
    """
    Base class for Tensorized Kernel Machines
    """

    @abstractmethod
    def __init__(
            self,
            features="poly",
            M: int = 8,
            R: int = 10,
            lengthscale: float = 0.5,
            l: float = 1e-5,
            num_sweeps: int = 10,
            key=random.PRNGKey(0),
            W_init=None,
            batch_size=None,
            dot_kron_method=None,
            loss=None,
            **kwargs,
    ):
        """
        Args:
            features (str): The feature map to use. Either 'poly' or 'fourier'. Defaults to 'poly'.
            M (int): The number of basis functions to use for the feature map or degree of the polynomial. Defaults to 8.
            R (int): The rank of the weight tensor. Defaults to 10.
            lengthscale (float): The lengthscale to use for the fourier feature map. Defaults to 0.5.
            l (float): The regularization strength. Defaults to 1e-5.
            num_sweeps (int): The number of sweeps to perform. Defaults to 10.
            key (jax.random.PRNGKey): The random key to use. Defaults to random.PRNGKey(0).
            W_init (jax.numpy.ndarray): The initial weight tensor. Defaults to None.
            batch_size (int): The batch size to use for the dot kron method.    Defaults to None.
            dot_kron_method (Callable): The method to use for the dot kron product. (NOT IMPLEMENTED YET)
            loss (Callable): The loss function to use. (NOT IMPLEMENTED YET)
        """
        self.M = M
        self.R = R
        self.l = l
        self.num_sweeps = num_sweeps
        self.lengthscale = lengthscale
        self.W_init = W_init
        self.key = key
        self.batch_size = batch_size
        self.features = features
        self.dot_kron_method = dot_kron_method
        self.loss = loss

    @abstractmethod
    def fit(self, X, y, **kwargs):
        return self

    @abstractmethod
    def predict(self, X):
        pass

    # @abstractmethod
    # def score(self, X, y):
    #     pass


class TensorizedKernelMachine(BaseTKM, BaseEstimator):
    """
    Tensorized Kernel Machine (TKM) for classification and regression.
    Implements fit, predict and score methods.
    """

    def __init__(self, *args, **kwargs):
        """
        Args:
            features (str): The feature map to use. Either 'poly' or 'fourier'. Defaults to 'poly'.
            M (int): The number of basis functions to use for the feature map or degree of the polynomial. Defaults to 8.
            R (int): The rank of the weight tensor. Defaults to 10.
            lengthscale (float): The lengthscale to use for the fourier feature map. Defaults to 0.5.
            l (float): The regularization strength. Defaults to 1e-5.
            num_sweeps (int): The number of sweeps to perform. Defaults to 10.
            key (jax.random.PRNGKey): The random key to use. Defaults to random.PRNGKey(0).
            W_init (jax.numpy.ndarray): The initial weight tensor. Defaults to None.
            batch_size (int): The batch size to use for the dot kron method.    Defaults to None.
            dot_kron_method (Callable): The method to use for the dot kron product. (NOT IMPLEMENTED YET)
            loss (Callable): The loss function to use. (NOT IMPLEMENTED YET)
        """
        super().__init__(*args, **kwargs)

    def _jit_fit_funs(self, X, y):
        """
        Jit the functions that are used in the fit method.
        """
        N, D = X.shape
        if self.features == "poly":
            self._features = jit(partial(polynomial, M=self.M, R=self.R))
        elif self.features == "fourier" or self.features == "rbf":
            self._features = jit(partial(fourier, M=self.M, R=self.R, lengthscale=self.lengthscale))

        dotkron = get_dotkron(batch_size=self.batch_size)
        self._dotkron = jit(dotkron)

        self._init_reg = jit(partial(_init_reg, W=self.W_init))  # TODO: check the memory footprint of partially
        # filling W
        self._init_g = jit(partial(_init_g, feature_func=self._features, W=self.W_init, X=X))  # TODO: check the memory
        self._fit_step = jit(partial(self._fit_step, X=X, y=y))
        self._sweep = jit(partial(self._sweep, D=D))
        # fit
        self._fit_w = jit(partial(self._fit_w, N=N, D=D), static_argnames=('N', 'D'))

        return self

    def _fit_step(self, d: int, value, X, y):
        """
        Perform a single step of the ALS algorithm to fit the model to the data.

        Args:
            d (int): The index of the current weight factor matrix.
            value (Tuple[jnp.ndarray, jnp.ndarray, float]): A tuple containing the current feature tensor G,
                the weight tensor W, and the regularization parameter reg.
            X (jnp.ndarray): The input data matrix of shape (n_samples, n_features).
            y (jnp.ndarray): The target values of shape (n_samples,).

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray, float]: A tuple containing the updated feature tensor G,
                the updated weight tensor W, and the updated regularization parameter reg.
        """
        (G, W, reg, loadings) = value  # TODO: jit(partial()) this
        phi_x = self._features(X[:, d])  # compute phi(x_d)
        G /= jnp.dot(phi_x, W[d])  # undoing the d-th element from G (contraction of all cores)
        CC, Cy = self._dotkron(phi_x, G, y)  # N by M_hat*R

        reg /= jnp.dot(W[d].T, W[d])  # regularization term
        regularization = self.l * jnp.kron(reg, jnp.eye(self.M))  # TODO: this results in sparse matrix,
        # check if multiplications with 0 need to be avoided

        w_d = jnp.linalg.solve((CC + regularization), Cy)  # solve systems of equation, least squares
        w_d = w_d.reshape((self.M, self.R), order='F')
        loadings = jnp.linalg.norm(w_d, ord=2, axis=0)
        w_d /= loadings
        W = W.at[d].set(w_d)

        reg *= jnp.dot(W[d].T, W[d])
        G *= jnp.dot(phi_x, W[d])

        return (G, W, reg, loadings)

    def _sweep(self, i, value, D):
        """
        Perform a sweep over the i-th factor matrix of a CP tensor of shape (I_1, ..., I_i, ..., I_D),
        apply the `fit_step` function to each factor matrix, using `value` as the initial value.

        Args:
            fit_step (Callable): The function to solve for a single factor matrix.
            i (int): The index of the sweep.
            value (jax.numpy.ndarray): The initial value to use for factor matrix.
            D (int): Dimension of the input data (i.e. X.shape[1]), this equal the number of factor matrices.

        Returns:
            jax.numpy.ndarray: The final value obtained after applying `fit_step` for each factor matrix.
        """
        return fori_loop(0, D, self._fit_step, init_val=value)

    def _fit_w(self, N: int, D: int):
        """
        Fit the model to the data. Function outputs the weights of the model, such that jit can be used.

        Args:
            N (int): The number of samples.
            D (int): The number of features.

        Returns:
            jnp.ndarray: The model weights.
        """
        W = self.W_init
        reg = jnp.ones((self.R, self.R))
        reg = fori_loop(0, D, self._init_reg, init_val=reg)
        G = jnp.ones((N, self.R))
        G = fori_loop(0, D, self._init_g, init_val=G)
        loadings = jnp.ones(self.R)

        (G, W, reg, loadings) = fori_loop(
            0, self.num_sweeps, self._sweep, init_val=(G, W, reg, loadings)
        )
        W.at[D].set(W[D]*loadings)
        return W

    def fit(
            self,
            X,
            y,
            **kwargs,
    ):
        """
        Fit the model to the data.

        Args:
            X (jnp.ndarray): The input data matrix of shape (n_samples, n_features).
            y (jnp.ndarray): The target values of shape (n_samples,).

        Returns:
            self: The fitted model.
        """
        N, D = X.shape
        self.D_ = D
        if self.W_init is None or self.W_init.shape != (D, self.M, self.R):
            W = random.normal(self.key, shape=(D, self.M, self.R))
            # W /= jnp.linalg.norm(W, axis=(1, 2), keepdims=True)
            self.W_init = W

        for d in range(D):
            temp = self.W_init[d] / jnp.linalg.norm(self.W_init[d], axis=0, ord=2)
            self.W_init = self.W_init.at[d].set(temp)

        self._jit_fit_funs(X=X, y=y)

        self.W_ = self._fit_w()

        return self

    @jit    # TODO: check if jit is necessary here
    def _predict_vmap(self, X):
        return vmap(lambda X, y: jnp.dot(self._features(X), y), (1, 0),)(X, self.W_).prod(0).sum(1)

    def predict(
            self,
            X,
            *args,
            **kwargs,
    ):
        """
        Predict the target values for the given input data.
        Args:
            X: input data (device array)
        Returns:
            device array with prediction scores for all classes (N,C)
                where N is the number of observations
                and C is the number of classes
        """

        return vmap(lambda X, y: jnp.dot(self._features(X), y), (1, 0),)(X, self.W_).prod(0).sum(1)

    # def score(self, X, y):
    #     """
    #     Args:
    #         X: input data (device array)
    #     Returns:
    #         device array with prediction scores based on RMSE
    #     """
    #     return rmse(y, self.predict(X))


class TensorizedKernelClassifier(TensorizedKernelMachine, ClassifierMixin):
    _estimator_type = "classifier"

    def __init__(
        self, 
        features="poly",
        M: int = 8,
        R: int = 10,
        lengthscale: float = 0.5,
        l: float = 1e-5,
        num_sweeps: int = 10,
        key=random.PRNGKey(0),
        W_init=None,
        batch_size=None,
        # _dotkron = get_dotkron(),
        **kwargs,
    ):
        super().__init__(
            features=features,
            M=M,
            R=R,
            lengthscale=lengthscale,
            l=l,
            num_sweeps=num_sweeps,
            key=key,
            W_init=W_init,
            batch_size=batch_size,
            **kwargs,
        )

    def fit(self, X, y, **kwargs):
        self.classes_ = jnp.unique(y)
        self.n_classes = len(self.classes_)
        super().fit(X, y, **kwargs)
        return self

    def decision_function(
        self,
        X,
        *args,
        **kwargs,
    ):
        """
        Args:
            X: input data (device array)
        Returns:
            device array with prediction scores for all classes (N,C)
                where N is the number of observations 
                and C is the number of classes
        """
        return super().predict(X)

    def predict(
        self,
        X,
        *args,
        **kwargs,
    ):
        """
        Args:
            x: input data (device array)
        Returns:
            device array with prediction scores for all classes (N,C)
                where N is the number of observations 
                and C is the number of classes
        """
        return jnp.sign(self.decision_function(X))

    # def score(self, X, y):
    #     """
    #     Args:
    #         X: input data (device array)
    #     Returns:
    #         device array with prediction scores for all classes (N,C)
    #             where N is the number of observations
    #             and C is the number of classes
    #     """
    #     return accuracy(y, self.predict(X))


class TensorizedKernelRegressor(TensorizedKernelMachine):
    _estimator_type = "regressor"

