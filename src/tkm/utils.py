import jax.numpy as jnp
from jax import jit
from jax import vmap


# @jit
def dotkron(a, b):
    """
    Row-wise kronecker product

    For column wise inspiration check khatri_rao: 
    https://scipy.github.io/devdocs/reference/generated/scipy.linalg.khatri_rao.html
    """
    # column-wise python
    # np.vstack([np.kron(a[:, k], b[:, k]) for k in range(b.shape[1])]).T

    # row-wise matlab
    # y = repmat(L,1,c2).*kron(R, ones(1, c1));
    # TODO check if transpose of repmat for second term option is: repmat() .* repmat()' 

    # TODO jax improvement for loops
    # TODO does python provide efficient kron() for vectors
    return jnp.vstack([jnp.kron(a[k, :], b[k, :]) for k in jnp.arange(b.shape[0])])


@jit
def vmap_dotkron(a,b):
    return vmap(jnp.kron)(a, b)


def batched_dotkron(A,B,y,batch_size=10000, **kwargs):
    N,DA = A.shape
    _,DB = B.shape
    CC = jnp.zeros((DA*DB,DA*DB))
    Cy = jnp.zeros((DA*DB,1))
    for n in range(0,N,batch_size):
        idx = min(n+batch_size-1,N)
        temp = vmap_dotkron(A[n:idx,:], B[n:idx,:]) # repmat(A(n:idx,:),1,DB)*kron(B(n:idx,:), ones(1, DA))
        CC += temp.T @ temp
        Cy += temp.T @ y[n:idx,:]
    return CC, Cy
