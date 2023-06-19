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


class TensorizedKernelMachine(object):
    def __init__(
        self, 
        features = polynomial,
        M: int = 8,
        R: int = 10,
        lengthscale: float = 0.5,  
        dotkron = get_dotkron(batch_size=None),
        **kwargs,
    ):
        """
        Args:
            features: kernel function for transforming input data
            M: M_hat degree of polynomials in one of the CP legs 
            R: rank
            lenghtscale: hyperparameters of fourierfeatures
            get_dotkron: function used for rowwise kronnecker product
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
        if W is None:
            W = random.normal(key, shape=(D,M,R)) if W is None else W
            W /= jnp.linalg.norm(W, axis=(1,2), keepdims=True)
        
        reg = jnp.ones((R,R))
        self.init_reg = jit(partial(self.init_reg, W=W)) #TODO: check the memory footprint of partially filling W
        reg = fori_loop(0, D, self.init_reg, init_val=reg)
        
        self.init_matd = jit(partial(self.init_matd, W=W, X=X)) #TODO: check the memory footprint of partially filling W
        Matd = jnp.ones((N,R))
        Matd = fori_loop(0, D, self.init_matd, init_val=Matd)

        self.fit_step = jit(partial(self.fit_step, X=X, y=y, l=l, M=M, R=R))
        self.sweep = jit(partial(self.sweep, D=D))
            
        (Matd, W, reg) = fori_loop(
            0, numberSweeps, self.sweep, init_val=(Matd, W, reg)
        )
        
        return W
    
    def init_reg(self, d, reg, W): # TODO: forloop is not necessary, should be able to do this with linalg
        reg *= jnp.dot(W[d].T, W[d])           # reg has shape R * R    
        return reg
    
    def init_matd(self, d, Matd, X, W):
        Mati = self.features(X[:,d])
        Matd *= jnp.dot(Mati, W[d])            # Matd has shape N * R, contraction of phi_x_d * w_d
        return Matd
    
    def sweep(self, i, value, D):
        return fori_loop(0, D, self.fit_step, init_val=value)
    
    def fit_step(self, d, value, l, M, R, X, y):
        (Matd, W, reg) = value # TODO: jit(partial()) this
        Mati = self.features(X[:,d])     # compute phi(x_d)                          
        Matd /= jnp.dot(Mati, W[d])     # undoing the d-th element from Matd (contraction of all cores)
        CC, Cy = self.dotkron(Mati,Matd,y)                                  # N by M_hat*R
        
        reg /= jnp.dot(W[d].T, W[d])                                    # regularization term
        regularization = l * jnp.kron(reg, jnp.eye(M)) # TODO: this results in sparse matrix, check if multiplications with 0 need to be avoided                
        
        x = jnp.linalg.solve((CC + regularization), Cy)         # solve systems of equation, least squares
        W = W.at[d].set( x.reshape((M,R)) )
        reg *= jnp.dot(W[d].T, W[d])
        Matd *= jnp.dot(Mati, W[d])

        return (Matd, W, reg)

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
