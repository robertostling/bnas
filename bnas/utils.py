"""Various utility functions."""

import theano.tensor as T

def expand_to_batch(x, batch_size, dim=-2):
    """Expand one dimension of `x` to `batch_size`."""
    return T.shape_padaxis(x, dim).repeat(batch_size, axis=dim)

