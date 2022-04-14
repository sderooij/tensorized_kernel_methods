import jax.numpy as jnp
from jax import random
# from jax.numpy.linalg import solve
# from jax.scipy.linalg import solve      # TODO check difference between the two
# from jax.scipy.linalg import kharti_rao

from tkm.features import polynomial, compile_feature_map
from tkm.utils import dotkron, vmap_dotkron
from jax import jit,vmap


def init(
    key,
    X,
    M: int = 8,
    R: int = 10,
):
    """
    Separating the fit function into initialization (init) and
     out the initialization from the 
    """
    N,D = X.shape #jnp.shape(X)
    W = random.normal(key, shape=(D,M,R))
    # list(range(D)) # TODO: JAX
    Matd = jnp.ones((N,R))
    reg = jnp.ones((R,R))
    
    # initializaiton of cores
    # intializing with the constant cores already contracted
    for d in range(D-1, -1, -1):
        # W[d] = 
        W.at[d].divide(jnp.linalg.norm(W[d]))                  # TODO: check if this is necessary
        reg *= (W[d].T @ W[d])           # reg has shape R * R
        Mati = polynomial(X[:,d])                       # TODO implement features function
        Matd *= Mati @ W[d]            # Matd has shape N * R, contraction of phi_x_d * w_d

    return W, reg, Matd



def fit( # TODO: type hinting
    key,
    X,
    y,
    M: int = 8,
    R: int = 10,
    l: float = 1e-5, # TODO: default value
    # lengthscale, # TODO: default value
    numberSweeps: int = 2, # TODO: default value

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

    polynomial = compile_feature_map(M=M)
    # polynomial_compiled = jit(partial(polynomial, M=M))

    N,D = X.shape #jnp.shape(X)
    W = random.normal(key, shape=(D,M,R))
    # list(range(D)) # TODO: JAX
    Matd = jnp.ones((N,R))
    reg = jnp.ones((R,R))
    
    # initializaiton of cores
    # intializing with the constant cores already contracted
    for d in range(D-1, -1, -1):
        # W[d] = 
        W.at[d].divide(jnp.linalg.norm(W[d]))                  # TODO: check if this is necessary
        reg *= jnp.dot(W[d].T, W[d])           # reg has shape R * R
        Mati = polynomial(X[:,d])
        Matd *= jnp.dot(Mati, W[d])            # Matd has shape N * R, contraction of phi_x_d * w_d

    # D,M,R = W.shape
    # itemax = numberSweeps * D # numberSweeps *(2*(D-1))+1;    # not necesarry in python
    # loss = []
    # error = []

    for s in range(numberSweeps):
        for d in range(D):
            # compute phi(x_d)
            Mati = polynomial(X[:,d])                               
            # undoing the d-th element from Matd (contraction of all cores)
            Matd /= jnp.dot(Mati, W[d])                                      
            C = vmap_dotkron(Mati,Matd)                                  # N by M_hat*R
            reg /= jnp.dot(W[d].T, W[d])                                    # regularization term
            regularization = l * jnp.kron(reg, jnp.eye(M)) # TODO: this results in sparse matrix, check if multiplications with 0 need to be avoided
            x = jnp.linalg.solve(                                   # solve systems of equations
                (jnp.dot(C.T, C) + regularization), 
                jnp.dot(C.T, y)
            )
            # print(loss(C,x,y,regularization))
            # print(error(C,x,y))
            # loss = jnp.linalg.norm(C @ x - y)**2 + x.T @ regularization @ x )  #TODO check if **2 is necessary (can it be done in function call of norm)
            # error =  jnp.mean(jnp.sign(C @ x) != y) # TODO not equal elementwise   # classification; for regression mean(((C*x)-y).^2)
            W.at[d].set( x.reshape((M,R)) )
            reg *= jnp.dot(W[d].T, W[d])
            Matd *= jnp.dot(Mati, W[d])

    return W #, loss, error


@jit
def loss(C,x,y,regularization):
    return jnp.linalg.norm(
        (jnp.dot(C,x) - y)**2 + 
        x.T.dot(regularization).dot(x)
    )


@jit
def error(C,x,y):
    return jnp.mean(jnp.sign(C.dot(x)) != y)


def predict(
    X, 
    W, 
    # hyperparameters,
):
    
    N, D = X.shape
    M = W[0].shape[0]
    # polynomial = compile_feature_map(M=M)
    score = jnp.ones((N,1))
    for d in range(D): #TODO JAX fori
        score *= jnp.dot(
            polynomial(X[:,d], M) , 
            W[d]
        )
    score = jnp.sum(score, 1)

    return score


def predict_vmap(
    X, 
    W, 
    # hyperparameters,
):
    
    M = W[0].shape[0]
    poly = compile_feature_map(M=M)

    return vmap(
        lambda x,y :jnp.dot(poly(x),y), (1,0),
    )(X, W).prod(0).sum(1)
