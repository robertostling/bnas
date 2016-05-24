"""Various utility functions."""

import theano.tensor as T

def expand_to_batch(x, batch_size, dim=-2):
    """Expand one dimension of `x` to `batch_size`."""
    return T.shape_padaxis(x, dim).repeat(batch_size, axis=dim)


def softmax_3d(x):
    """Generalization of T.nnet.softmax for 3D tensors"""
    return T.nnet.softmax(x.reshape(
        (x.shape[0]*x.shape[1], x.shape[2]))).reshape(
        (x.shape[0], x.shape[1], x.shape[2]))

