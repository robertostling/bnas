from theano import tensor as T

def expand_to_batch(x, batch_size, dim=-2):
    return T.shape_padaxis(x, dim).repeat(batch_size, axis=dim)

