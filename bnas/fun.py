"""Wrapper for creating Theano functions for training/inference mode"""

import theano
from theano import tensor as T

# 1 = training mode
# 0 = inference mode
train_mode = T.bscalar('train_mode')

def function(inputs=[], outputs=[], default_mode=0, **kwargs):
    use_train_mode = train_mode in theano.gof.graph.ancestors(inputs)
    extra_args = [train_mode] if use_train_mode else []

    f = theano.function(
            list(inputs)+extra_args,
            outputs,
            on_unused_input='raise',
            **kwargs)
    def g(*args):
        if default_mode is None:
            # args[-1] is the train_mode value
            if use_train_mode:
                # f() includes train_mode, pass arguments directly
                return f(*args)
            else:
                # f() does not include train_mode, drop the last argument
                return f(*(args[:-1]))
        else:
            # no train_mode value in args
            if use_train_mode:
                # use the default value of train_mode
                return f(*(args + (default_mode,)))
            else:
                # f() does not include train_mode, pass arguments directly
                return f(*args)
    return g

