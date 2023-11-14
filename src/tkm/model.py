from functools import partial

import jax.numpy as jnp
from jax import random
from jax.lax import fori_loop
# from jax.numpy.linalg import solve
# from jax.scipy.linalg import solve      # TODO check difference between the two
# from jax.scipy.linalg import kharti_rao

from tkm.features import polynomial, fourier
from tkm.kron import get_dotkron
from jax import jit,vmap

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
import tensorly as tl
tl.set_backend("jax")


class TensorizedKernelMachine(BaseEstimator, ClassifierMixin):
    def __init__(
        self, 
        features="poly",
        M: int = 8,
        R: int = 10,
        lengthscale: float = 0.5,
        l: float = 1e-5,
        numberSweeps: int = 10,
        key=random.PRNGKey(0),
        W_init=None,
        batch_size=None,
        # _dotkron = get_dotkron(),
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
        self.M = M
        self.R = R
        self.l = l
        self.numberSweeps = numberSweeps
        self.lengthscale = lengthscale
        self.W_init = W_init
        self.key = key
        self.batch_size = batch_size
        self.features = features
        # self._dotkron = jit(get_dotkron(batch_size=batch_size))
        # self.features = jit(partial(features, M=M, R=R, lengthscale=lengthscale, **kwargs))
        # self._fit_w = jit(partial(self._fit_w))
        # self.predict = jit(partial(self.predict, **kwargs))
        
    def _jit_funcs(self, x, y):
        if self.features == "poly":
            self._features = jit(partial(polynomial, M=self.M, R=self.R))
        elif self.features == "fourier":
            self._features = jit(partial(fourier, M=self.M, R=self.R, lengthscale=self.lengthscale))

        self._dotkron = jit(get_dotkron(batch_size=self.batch_size))
        self._fit_w = jit(partial(self._fit_w))
        self.decision_function = jit(partial(self.decision_function))

        self._init_reg = jit(partial(self._init_reg, W=self.W_init))  # TODO: check the memory footprint of partially
        # filling W
        self._init_g = jit(partial(self._init_g, W=self.W_init, X=x))  # TODO: check the memory footprint of partially
        self._fit_step = jit(partial(self._fit_step, X=x, y=y))
        self._sweep = jit(partial(self._sweep, D=self.D_))

        return self
    
    def _init_reg(self, d, reg, W): # TODO: forloop is not necessary, should be able to do this with linalg
        """
        Computes the regularization term for the d-th factor matrix of the weight tensor.

        Args:
            d (int): The factor matrix of the weight tensor to compute the regularization term for.
            reg (float): The regularization strength.
            W (jax.interpreters.xla.DeviceArray): The weight tensor.

        Returns:
            jax.interpreters.xla.DeviceArray: The regularization term for the d-th factor matrix of the weight tensor.
        """
        reg *= jnp.dot(W[d].T, W[d])           # reg has shape R * R    
        return reg
    
    def _init_g(self, d, G, X, W):
        """
        Initializes the G matrix for a given dimension d.

        Args:
            d (int): The dimension for which to initialize Matd.
            G (jax.numpy.ndarray): The G matrix to initialize.
            X (jax.numpy.ndarray): The input data matrix of shape (N, D).
            W (jax.numpy.ndarray): The weight matrix of shape (D, R).

        Returns:
            jax.numpy.ndarray: The initialized Matd matrix of shape (N, R).
        """
        phi_x = self._features(X[:,d])
        G *= jnp.dot(phi_x, W[d])            # G has shape N * R, contraction of phi_x_d * w_d
        return G
    
    def _sweep(self, i, value, D):
        """
        Perform a sweep over the i-th factor matrix of a CP tensor of shape (I_1, ..., I_i, ..., I_D),
        apply the `fit_step` function to each factor matrix, using `value` as the initial value.

        Args:
            i (int): The index of the dimension to sweep over.
            value (jax.numpy.ndarray): The initial value to use for factor matrix.
            D (int): The size of the dimension to sweep over.

        Returns:
            jax.numpy.ndarray: The final value obtained after applying `fit_step` for each factor matrix.
        """
        return fori_loop(0, D, self._fit_step, init_val=value)
    
    def _fit_step(self, d, value, X, y):
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
        (G, W, reg) = value # TODO: jit(partial()) this
        phi_x = self._features(X[:,d])     # compute phi(x_d)
        G /= jnp.dot(phi_x, W[d])     # undoing the d-th element from G (contraction of all cores)
        CC, Cy = self._dotkron(phi_x, G, y)                                  # N by M_hat*R
        
        reg /= jnp.dot(W[d].T, W[d])                                    # regularization term
        regularization = self.l * jnp.kron(reg, jnp.eye(self.M)) # TODO: this results in sparse matrix,
        # check if multiplications with 0 need to be avoided
        
        w_d = jnp.linalg.solve((CC + regularization), Cy)         # solve systems of equation, least squares
        W = W.at[d].set(w_d.reshape((self.M, self.R), order='F'))
        reg *= jnp.dot(W[d].T, W[d])
        G *= jnp.dot(phi_x, W[d])

        return (G, W, reg)
    
    def _fit_w(self, x, y):
        """
        Fit the model to the data. Function outputs the weights of the model, such that jit can be used.

        Args:
            x (jnp.ndarray): The input data matrix of shape (n_samples, n_features).
            y (jnp.ndarray): The target values of shape (n_samples,).

        Returns:
            jnp.ndarray: The model weights.
        """
        N,D = x.shape
        W = self.W_init
        reg = jnp.ones((self.R, self.R))
        reg = fori_loop(0, D, self._init_reg, init_val=reg)
        G = jnp.ones((N, self.R))
        G = fori_loop(0, D, self._init_g, init_val=G)
            
        (G, W, reg) = fori_loop(
            0, self.numberSweeps, self._sweep, init_val=(G, W, reg)
        )
        
        return W
    
    def fit(
        self,
        x,
        y,
        **kwargs,
    ):
        """
        Fit the model to the data. 

        Args:
            x (jnp.ndarray): The input data matrix of shape (n_samples, n_features).
            y (jnp.ndarray): The target values of shape (n_samples,).

        Returns:
            self: The fitted model.
        """
        N, D = x.shape
        self.D_ = D
        if self.W_init is None:
            W = random.normal(self.key, shape=(D, self.M, self.R))
            W /= jnp.linalg.norm(W, axis=(1, 2), keepdims=True)
            self.W_init = W
        self._jit_funcs(x=x, y=y)
        self.classes_ = jnp.unique(y)
        self.n_classes_ = len(self.classes_)


        self.W_ = self._fit_w(x=x, y=y)
        
        return self

    def decision_function(
        self,
        x,
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
        return vmap(
            lambda x,y:jnp.dot(self.features(x),y), (1,0),
        )(x, self.W_).prod(0).sum(1)

    def predict(
        self,
        x,
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
        return jnp.sign(self.decision_function(x))
