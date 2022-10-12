from functools import partial
import jax.numpy as jnp
from jax import random
# from jax.numpy.linalg import solve
# from jax.scipy.linalg import solve      # TODO check difference between the two
# from jax.scipy.linalg import kharti_rao

from tkm.features import polynomial, fourier
from tkm.kron import get_dotkron
from jax import jit,vmap


class TensorizedKernelMachine(object):
    def __init__(
        self, 
        dotkron = get_dotkron(batch_size=None),
        features = polynomial,
        M: int = 8,
        R: int = 10,
        lengthscale: float = 0.5,  
        **kwargs,
    ):
        """
        Args:
            dotkron: function used for rowwise kronnecker product
            features: kernel function for transforming input data
            M: M_hat degree of polynomials in one of the CP legs 
            R: rank
            lenghtscale: hyperparameters of fourierfeatures
            batch_size: if is not None batched_dotkron will be used
        """

        self.dotkron = jit(dotkron)
        self.features = jit(partial(features, M=M, R=R, lengthscale=lengthscale, **kwargs))
        self.fit = jit(partial(self.fit, M=M, R=R, **kwargs))
        self.predict = jit(partial(self.predict, **kwargs))

    def fit(
        self,
        key,
        X,
        y,
        l: float = 1e-5,
        numberSweeps: int = 10,
        M: int = 8,
        R: int = 10,
        W = None,
        **kwargs,
    ):
        """
        Args:
            X: data
            y: labels
            l: peanalty term of regularisation (denoted by lambda)
            numberSweeps: number of ALS sweeps, 1 -> D -> 1 (not including last one)
                1 and D are covered once
                middle is covered twice
                alternative is doing linear pas 1,...,D (not in this code)
            M: M_hat degree of polynomials in one of the CP legs 
            R: rank
            W: device array of weight, used to bypass random init

        Returns:
            weights: device array
        """

        N,D = X.shape
        W = random.normal(key, shape=(D,M,R)) if W is None else W
        Matd = jnp.ones((N,R))
        reg = jnp.ones((R,R))
        
        # initializaiton of cores
        # intializing with the constant cores already contracted
        for d in range(D-1, -1, -1):
            W = W.at[d].divide(jnp.linalg.norm(W[d])) if W is None else W       # TODO: check if this is necessary
            reg *= jnp.dot(W[d].T, W[d])           # reg has shape R * R
            Mati = self.features(X[:,d])
            Matd *= jnp.dot(Mati, W[d])            # Matd has shape N * R, contraction of phi_x_d * w_d

        for s in range(numberSweeps): #TODO fori jax loop
            for d in range(D): #TODO fori jax loop
                Mati = self.features(X[:,d])     # compute phi(x_d)                          
                Matd /= jnp.dot(Mati, W[d])     # undoing the d-th element from Matd (contraction of all cores)
                CC, Cy = self.dotkron(Mati,Matd,y)                                  # N by M_hat*R
                
                reg /= jnp.dot(W[d].T, W[d])                                    # regularization term
                regularization = l * jnp.kron(reg, jnp.eye(M)) # TODO: this results in sparse matrix, check if multiplications with 0 need to be avoided                
                
                x = jnp.linalg.solve((CC + regularization), Cy)         # solve systems of equation
                W = W.at[d].set( x.reshape((M,R)) )
                reg *= jnp.dot(W[d].T, W[d])
                Matd *= jnp.dot(Mati, W[d])
        
        return W

    def predict(
        self,
        X, 
        W,
        *args,**kwargs,
    ):
        """
        Args:
            X: input data (device array)
            W: weights (device array)
        Returns:
            device array with prediction scores for all classes (N,C)
                where N is the number of observations 
                and C is the number of classes
        """
        return vmap(
            lambda x,y :jnp.dot(self.features(x),y), (1,0),
        )(X, W).prod(0).sum(1)
