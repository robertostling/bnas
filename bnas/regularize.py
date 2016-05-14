"""Regularizations.

Each regularization method is implemented as a subclass of
:class:`Regularizer`,
where the constructor takes the hyperparameters, and the `__call__` method
constructs the symbolic loss expression given a parameter.

These are made for use with :meth:`Model.regularize`, but can also be used
directly in the :meth:`loss` method of :class:`.Model` subclasses.
"""

import theano
import theano.tensor as T
from theano.ifelse import ifelse

class Regularizer: pass

class L2(Regularizer):
    """L2 loss."""

    def __init__(self, penalty=0.01):
        self.penalty = penalty

    def __call__(self, p):
        return T.sqrt(T.sqr(p).sum()) * T.as_tensor_variable(self.penalty)


class StateNorm(Regularizer):
    """Squared norm difference between recurrent states.

    Note that this method seems to be unstable if the initial hidden state is
    initialized to zero.

    David Krueger & Roland Memisevic (2016).
    `Regularizing RNNs by stabilizing activations. <http://arxiv.org/pdf/1511.08400v7.pdf>`_
    """
    def __init__(self, penalty=50.0):
        self.penalty = penalty

    def __call__(self, p, p_mask):
        """Compute the squared norm difference of a sequence.

        Example
        -------
        >>> def loss(self, outputs, outputs_mask):
        ...     # loss() definition from a custom Model subclass
        ...     loss = super().loss()
        ...     pred_states, pred_symbols = self(outputs, outputs_mask)
        ...     # Include transition from initial state
        ...     pred_states = T.concatenate([initial_state, pred_states],
        ...                                 axis=0)
        ...     return loss + StateNorm(50.0)(pred_states, outputs_mask)
        """
        mask = p_mask[:-1]
        l2 = T.sqrt(T.sqr(p).sum(axis=2))
        diff = (l2[1:] - l2[:-1]) * mask
        return (self.penalty * T.sqr(diff).sum() /
                mask.sum().astype(theano.config.floatX))

