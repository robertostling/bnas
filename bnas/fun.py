"""Wrapper for creating Theano functions for training/inference mode"""

import theano
from theano import tensor as T

# 1 = training mode
# 0 = inference mode
train_mode = T.bscalar('train_mode')

def function(inputs=[], outputs=[], default_mode=0, **kwargs):
    f = theano.function(
            list(inputs)+[train_mode],
            outputs,
            on_unused_input='warn',
            **kwargs)
    if default_mode is None:
        return f
    def g(*args):
        return f(*(args + (default_mode,)))
    return g

