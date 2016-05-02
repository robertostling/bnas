"""Regularizations.

Each regularization method is implemented as a subclass of `Regularizer`,
where the constructor takes the hyperparameters, and the `__call__` method
constructs the symbolic loss expression given a parameter.
"""

import theano.tensor as T
from theano.ifelse import ifelse

class Regularizer: pass

class L2(Regularizer):
    """L2 loss."""

    def __init__(self, penalty=0.01):
        self.penalty = penalty

    def __call__(self, p):
        return T.sqrt(T.sqr(p).sum()) * T.as_tensor_variable(self.penalty)


def seq_l2(x):
    """Compute L2 norms over a sequence of batches."""
    return T.sqrt(T.sqr(x).sum(axis=2))


class StateNorm(Regularizer):
    """Squared norm difference between recurrent states.

    For sequences of less than two elements, the value is defined as 0.

    TODO: is this wise, or do we encourage empty sequences?
   
    David Krueger & Roland Memisevic (2016).
    Regularizing RNNs by stabilizing activations.
    http://arxiv.org/pdf/1511.08400v7.pdf
    """
    def __init__(self, penalty=50.0):
        self.penalty = penalty

    def __call__(self, p):
        return ifelse(
                p.shape[0] >= 2,
                      T.sqr(seq_l2(p[1:,:,:]) - seq_l2(p[:-1,:,:])).mean()
                    * T.as_tensor_variable(self.penalty),
                    T.as_tensor_variable(0.0))

