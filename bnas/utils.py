"""Various utility functions."""

import theano
import theano.tensor as T

def expand_to_batch(x, batch_size, dim=-2):
    """Expand one dimension of `x` to `batch_size`."""
    return T.shape_padaxis(x, dim).repeat(batch_size, axis=dim)

def softmax_masked(x, mask):
    """Softmax over a batch of masked items.

    x : matrix
        Softmax will be computed over the rows of this matrix.
    mask : matrix
        Binary matrix, zero elements will be zeroed out before normalization.
    """
    # Adapted from Theano's reference implementation:
    # http://deeplearning.net/software/theano/library/tensor/nnet/nnet.html#theano.tensor.nnet.nnet.softmax
    e_x = T.exp(x - x.max(axis=1, keepdims=True)).astype(
            theano.config.floatX) * mask.astype(theano.config.floatX)
    return e_x / e_x.sum(axis=1, keepdims=True)

def softmax_3d(x):
    """Generalization of T.nnet.softmax for 3D tensors"""
    return T.nnet.softmax(x.reshape(
        (x.shape[0]*x.shape[1], x.shape[2]))).reshape(
        (x.shape[0], x.shape[1], x.shape[2]))

def softmax_4d(x):
    """Generalization of T.nnet.softmax for 4D tensors"""
    return T.nnet.softmax(x.reshape(
        (x.shape[0]*x.shape[1]*x.shape[2], x.shape[3]))).reshape(
        (x.shape[0], x.shape[1], x.shape[2], x.shape[3]))

def concatenate(tensor_list, axis=0):
    """
    From https://github.com/nyu-dl/dl4mt-tutorial/
    Not sure if Theano still has this issue, seems to decrease performance in
    fast_compile mode at least.

    Alternative implementation of `theano.tensor.concatenate`.
    This function does exactly the same thing, but contrary to Theano's own
    implementation, the gradient is implemented on the GPU.
    Backpropagating through `theano.tensor.concatenate` yields slowdowns
    because the inverse operation (splitting) needs to be done on the CPU.
    This implementation does not have that problem.
    :usage:
        >>> x, y = theano.tensor.matrices('x', 'y')
        >>> c = concatenate([x, y], axis=1)
    :parameters:
        - tensor_list : list
            list of Theano tensor expressions that should be concatenated.
        - axis : int
            the tensors will be joined along this axis.
    :returns:
        - out : tensor
            the concatenated tensor expression.
    """
    assert len({tt.ndim for tt in tensor_list}) == 1
    axis = axis % tensor_list[0].ndim
    concat_size = sum(tt.shape[axis] for tt in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = T.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tt.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = T.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]

    return out

