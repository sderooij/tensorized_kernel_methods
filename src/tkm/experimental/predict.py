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