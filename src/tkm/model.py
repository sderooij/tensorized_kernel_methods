import jax.numpy as jnp
from jax import random
from jax.numpy.linalg import solve
# from jax.scipy.linalg import solve      # TODO check difference between the two
from jax.scipy.linalg import kharti_rao

from tkm.features import polynomial as features
from tkm.utils import dotkron

key = random.PRNGKey(42)


def fit( # TODO: type hinting
    X,
    y, 
    M, # TODO: default value
    R, # TODO: default value
    l, # TODO: default value
    lengthscale, # TODO: default value
    numberSweeps, # TODO: default value
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

    _, D = X.shape() #jnp.shape(X)
    W = list(range(D))
    Matd = 1
    reg = 1
    
    # initializaiton of cores
    # intializing with the constant cores already contracted
    for d in range(D-1, -1, -1):
        W[d] = random.normal(key, shape=(M,R))
        W[d] /= jnp.norm(W[d])                  # TODO: check if this is necessary
        reg = reg * (W[d].T.matmul(W[d]))           # reg has shape R * R
        Mati = features()                       # TODO implement features fucntion
        Matd = Mati.matmul(W[d]) * Matd             # Matd has shape N * R, contraction of phi_x_d * w_d



    itemax = numberSweeps * D # numberSweeps *(2*(D-1))+1;    # not necesarry in python
    loss = jnp.zeros(itemax,1)
    error = jnp.zeros(itemax,1)

    for ite in range(itemax):

        ''' irrelevant for python
        loopind = mod(ite-1,2*(D-1))+1;
        if loopind <= D
            d = loopind;
        else
            d = 2*D-loopind;
        end
        '''

        Mati = features()                               # compute phi(x_d)   #TODO implement function
        
        reg /= W[d].T.matmul(W[d])                      # regularization term
        Matd /= Mati.matmul(W[d])                       # undoing the d-th element from 
        C = dotkron(Mati,Matd)                          # N by M_hat*R       #TODO implement function
        regularization = l * jnp.kron(reg, jnp.eye(M))

        regularization = l * jnp.kron(reg, jnp.eye(M))
        x = solve(                                      # solve systems of equations
            (C.T.matmul(C) + regularization), 
            (C.T.matmul(y)) 
        )
        loss[ite] = jnp.norm(C.matmul(x)-y)**2 + x.T.matmul(regularization).matmul(x) #TODO check if **2 is necessary (can it be done in function call of norm)
        error[ite] = jnp.mean(jnp.sign(C.matmul(x)) != y)  # TODO not equal elementwise                  # classification; for regression mean(((C*x)-y).^2)
        W[d] = jnp.reshape(x,shape=(M,R))
        reg *= W[d].T.matmul(W[d])
        Matd *= Mati*W[d]


    return W, loss, error
