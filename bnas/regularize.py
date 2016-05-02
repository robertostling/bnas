"""Regularizations."""

class Regularizer: pass

class L2(Regularizer):
    """L2 loss."""

    def __init__(self, penalty=0.01):
        self.penalty = penalty

    def __call__(self, p):
        return T.sqrt(T.sqr(p).sum()) * self.penalty


def seq_l2(x):
    """Compute L2 norms over a sequence of batches."""
    return T.sqrt(T.sqr(x).sum(axis=2))


class StateNorm(Regularizer):
    """Squared norm difference between recurrent states.
   
    David Krueger & Roland Memisevic (2016).
    Regularizing RNNs by stabilizing activations.
    http://arxiv.org/pdf/1511.08400v7.pdf
    """
    def __init__(self, penalty=50.0):
        self.penalty = penalty

    def __call__(self p):
        # TODO: what about sequences of length < 2?
        return   T.sqr(seq_l2(p[1:,:,:]) - seq_l2(p[:-1,:,:])).mean() \
               * self.penalty

