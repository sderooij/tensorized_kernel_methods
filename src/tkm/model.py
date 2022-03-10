import jax.numpy as jnp
from jax import random
# from jax.numpy.linalg import solve
# from jax.scipy.linalg import solve      # TODO check difference between the two
# from jax.scipy.linalg import kharti_rao

from tkm.features import polynomial
from tkm.utils import dotkron

key = random.PRNGKey(42)


def fit( # TODO: type hinting
    X,
    y, 
    M: int = 8, # TODO: default value
    R: int = 10, # TODO: default value
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

    _, D = X.shape #jnp.shape(X)
    W = list(range(D))
    Matd = 1
    reg = 1
    
    # initializaiton of cores
    # intializing with the constant cores already contracted
    for d in range(D-1, -1, -1):
        W[d] = random.normal(key, shape=(M,R))
        W[d] /= jnp.linalg.norm(W[d])                  # TODO: check if this is necessary
        reg = reg * (W[d].T @ W[d])           # reg has shape R * R
        Mati = polynomial(X[:,d], M)                       # TODO implement features fucntion
        Matd = Mati @ W[d] * Matd             # Matd has shape N * R, contraction of phi_x_d * w_d

    itemax = numberSweeps * D # numberSweeps *(2*(D-1))+1;    # not necesarry in python
    loss = []
    error = []

    for ite in range(itemax):

        ''' irrelevant for python
        loopind = mod(ite-1,2*(D-1))+1;
        if loopind <= D
            d = loopind;
        else
            d = 2*D-loopind;
        end
        '''

        Mati = polynomial(X[:,d], M)                               # compute phi(x_d)
        
        reg /= W[d].T @ W[d]                                    # regularization term
        Matd /= Mati @ W[d]                                     # undoing the d-th element from 
        C = dotkron(Mati,Matd)                                  # N by M_hat*R
        regularization = l * jnp.kron(reg, jnp.eye(M))

        regularization = l * jnp.kron(reg, jnp.eye(M))
        x = jnp.linalg.solve(                                   # solve systems of equations
            (C.T @ C + regularization), 
            (C.T @ y) 
        )
        loss.append( jnp.linalg.norm(C @ x - y)**2 + x.T @ regularization @ x )   #TODO check if **2 is necessary (can it be done in function call of norm)
        error.append( jnp.mean(jnp.sign(C @ x) != y) ) # TODO not equal elementwise   # classification; for regression mean(((C*x)-y).^2)
        W[d] = x.reshape((M,R))
        reg *= W[d].T @ W[d]
        Matd *= Mati @ W[d]

    return W, loss, error
