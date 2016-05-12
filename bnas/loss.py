"""Functions for computing loss functions.

Instances of :class:`Model` typically define a `loss()` method using using
model-specific values. This module contains helper functions that could be
useful.

For most loss functions it is enough to use pure Theano, e.g. using
:func:`theano.tensor.nnet.categorical_crossentropy` or just
``T.sqr(prediction-target).sum()``.

See for instance `examples/rnn.py` for an example.
"""

import theano
import theano.tensor as T

def batch_sequence_crossentropy(x, target, target_mask):
    """Compute the mean categorical cross-entropy over a sequence of batches.

    Parameters
    ----------
    x : tensor3
        Symbolic variable of shape (sequence_length, batch_size, n_symbols).
    target : lmatrix
        Symbolic variable of shape (sequence_length, batch_size).
    target_mask : matrix
        Binary mask for `target`, also of shape (sequence_length, batch_size).

    Returns
    -------
    cross_entropy : Theano symbolic expression
        Symbolic expression for the mean categorical cross-entropy per
        sequence over the batches, using non-masked entries only.
    """
    batch_size = x.shape[1].astype(theano.config.floatX)
    return (T.nnet.categorical_crossentropy(
                x.reshape((x.shape[0]*x.shape[1], x.shape[2])),
                target.flatten())
            * target_mask.flatten()).sum() / batch_size

