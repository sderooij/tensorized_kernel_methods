from functools import partial
import jax.numpy as jnp
from jax import random
# from jax.numpy.linalg import solve
# from jax.scipy.linalg import solve      # TODO check difference between the two
# from jax.scipy.linalg import kharti_rao

from tkm.features import polynomial, compile_feature_map, fourier
from tkm.utils import dotkron, vmap_dotkron, batched_dotkron, vmap_dotkron_new
from jax import jit,vmap
# import jmp


def dotkron(batch_size=None):
    if batch_size is None:
        return vmap_dotkron_new # TODO: should have the right return types
    else:
        return partial(batched_dotkron, batch_size=batch_size)


class TensorizedKernelMachine(object):
    def __init__(
        self, 
        dotkron = dotkron(batch_size=None),
        policy = None,
        **kwargs,
    ):
        # Fine for JAX since this is a constant.
        # self.policy = policy if policy is not None else jmp.Policy(...)
        
        # This is equivalent to `jax.jit(partial(Experiment.forward, self))`.

        self.dotkron = jit(dotkron)

        self.fit = jit(partial(self.fit, **kwargs))
        self.predict = jit(partial(self.predict_vmap, **kwargs))

    def fit(
        self,
        key,
        X,
        y,
        M: int = 8,
        R: int = 10,
        l: float = 1e-5,
        lengthscale: float = 0.5,
        numberSweeps: int = 10,
        feature_map=polynomial,
        W = None,
        **kwargs,
    ):
        """
        input
            X: data
            y: labels
            M: M_hat degree of polynomials in one of the CP legs 
            R: rank
            l: peanalty term of regularisation (denoted by lambda)
            lenghtscale: hyperparameters of features
            numberSweeps: number of ALS sweeps, 1 -> D -> 1 (not including last one)
                1 and D are covered once
                middle is covered twice
                alternative is doing linear pas 1,...,D (not in this code)

        returns
            weights
            loss
            error: list of errors per ALS step
        """
        # TODO: jmp policy
        # my_policy = self.policy
        # params, x = my_policy.cast_to_compute((params, x))

        # feature_map = feature_list[feature_idx]

        features = compile_feature_map(feature_map, M=M, lengthscale=lengthscale)
        # polynomial_compiled = jit(partial(polynomial, M=M))

        N,D = X.shape #jnp.shape(X)
        W = random.normal(key, shape=(D,M,R)) if W is None else W
        # list(range(D)) # TODO: JAX
        Matd = jnp.ones((N,R))
        reg = jnp.ones((R,R))
        
        # initializaiton of cores
        # intializing with the constant cores already contracted
        for d in range(D-1, -1, -1):
            W = W.at[d].divide(jnp.linalg.norm(W[d])) if W is None else W       # TODO: check if this is necessary
            reg *= jnp.dot(W[d].T, W[d])           # reg has shape R * R
            Mati = features(X[:,d])
            Matd *= jnp.dot(Mati, W[d])            # Matd has shape N * R, contraction of phi_x_d * w_d

        # D,M,R = W.shape
        # itemax = numberSweeps * D # numberSweeps *(2*(D-1))+1;    # not necesarry in python
        # loss = []
        # error = []
        # i=0
        for s in range(numberSweeps): #TODO fori jax loop
            for d in range(D): #TODO fori jax loop
                # compute phi(x_d)
                Mati = features(X[:,d])                               
                # undoing the d-th element from Matd (contraction of all cores)
                Matd /= jnp.dot(Mati, W[d])                                      
                
                #TODO: reformat CC
                C = vmap_dotkron(Mati,Matd)                                  # N by M_hat*R
                
                reg /= jnp.dot(W[d].T, W[d])                                    # regularization term
                regularization = l * jnp.kron(reg, jnp.eye(M)) # TODO: this results in sparse matrix, check if multiplications with 0 need to be avoided
                
                #TODO: reformat cc
                x = jnp.linalg.solve(                                   # solve systems of equations
                    (jnp.dot(C.T, C) + regularization), 
                    jnp.dot(C.T, y)
                )
                # loss.append(float(loss_function(C,x,y,regularization)[0][0]))
                # print(error(C,x,y))
                # loss = jnp.linalg.norm(C @ x - y)**2 + x.T @ regularization @ x )  #TODO check if **2 is necessary (can it be done in function call of norm)
                # error =  jnp.mean(jnp.sign(C @ x) != y) # TODO not equal elementwise   # classification; for regression mean(((C*x)-y).^2)
                W = W.at[d].set( x.reshape((M,R)) )
                reg *= jnp.dot(W[d].T, W[d])
                Matd *= jnp.dot(Mati, W[d])
                # i+=1
        
        # TODO: my_policy.cast_to_output(y)
        return W #, loss #error 

    def predict(
        self,
        X, 
        W, 
        # hyperparameters,
        feature_map=polynomial,
        *args,**kwargs,
    ):
        features = compile_feature_map(feature_map, *args,**kwargs)
        N, D = X.shape
        M = W[0].shape[0]
        # polynomial = compile_feature_map(M=M)
        score = jnp.ones((N,1))
        for d in range(D): #TODO JAX fori
            score *= jnp.dot(
                features(X[:,d]) , 
                W[d]
            )
        score = jnp.sum(score, 1)

        return score


    def predict_vmap(
        self,
        X, 
        W, 
        feature_map=polynomial,
        *args,**kwargs,
    ):
        
        # M = W[0].shape[0]
        features = compile_feature_map(feature_map, *args,**kwargs)

        return vmap(
            lambda x,y :jnp.dot(features(x),y), (1,0),
        )(X, W).prod(0).sum(1)


@jit
def loss_function(C,x,y,regularization):
    return jnp.power(jnp.linalg.norm(jnp.dot(C,x) - y),2) + x.T.dot(regularization).dot(x)


@jit
def error(C,x,y):
    return jnp.mean(jnp.sign(C.dot(x)) != y)
