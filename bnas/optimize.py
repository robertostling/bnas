"""Optimization algorithms.

This module provides different algorithms for optimization through (typically)
stochastic mini-batch gradient descent.
"""

import random
from collections import OrderedDict
import pickle

import numpy as np
import theano
from theano import tensor as T

from .fun import function


def iterate_batches(data, batch_size, len_f=None, n_batches=16):
    """Iterate over minibatches.

    Arguments
    ---------
    data : list of data items (typically example/label pairs)
        Data set to iterate over
    batch_size : int
        Minibatch size. If len(data) is at above this, each batch is
        guaranteed to be of exactly size batch_size.
    len_f : function
        If this is defined, it should be a function mapping items from the
        data array to some ordered type. n_batches will be randomly
        sampled at a time, the examples inside sorted and cut up into batches.
        This is useful for variable-length sequences, so that batches aren't
        too sparse.
    n_batches : int
    """
    order = list(range(len(data)))
    random.shuffle(order)
    if len(data) <= batch_size:
        yield data
    elif len_f is None:
        for i in range(0, len(data) - len(data)%batch_size, batch_size):
            yield [data[j] for j in order[i:i+batch_size]]
    else:
        for i in range(0, len(data), batch_size*n_batches):
            if i > len(data) - batch_size: return
            subset = [data[j] for j in order[i:i+batch_size*n_batches]]
            subset.sort(key=len_f)
            useful_length = len(subset) - len(subset)%batch_size
            for j in range(0, useful_length, batch_size):
                yield subset[j:j+batch_size]


class Optimizer:
    """Base class for optimizers.

    Arguments
    ---------
    params : iterable over (name, parameter) tuples
        Parameters to optimize, in simple cases it's enough to pass
        Model.parameters().
    loss : Theano symbolic expression
        Loss function to minimize.
    inputs : list of Theano variables
        Inputs to the model to optimize
    outputs : list of Theano variables
        Outputs of the model to optimize, the loss should depend on
        `inputs + outputs`.
    grad_max_norm : float
        Clip gradients at this value.
    """
    def __init__(self, params, loss, inputs=[], outputs=[], grad_max_norm=None):
        self.params = OrderedDict(('_'.join(name), p) for name, p in params)
        self.loss = loss
        self.inputs = inputs
        self.outputs = outputs
        self.grad_max_norm = grad_max_norm
        self._grad_fun = None
        self.optimizer_params = []
        self.n_updates = 0

        self.raw_grad = OrderedDict((name, T.grad(loss, param))
                                    for name, param in self.params.items())
        if grad_max_norm is None:
            self.grad = self.raw_grad
        else:
            norm = T.sqrt(T.stack(
                [T.sqr(g).sum() for g in self.raw_grad.values()],
                axis=0).sum())
            a = T.switch(norm < self.grad_max_norm, 1, self.grad_max_norm/norm)
            self.grad = OrderedDict((name, a*g)
                                    for name, g in self.raw_grad.items())

    def shared(self, *args, **kwargs):
        s = theano.shared(*args, **kwargs)
        self.optimizer_params.append(s)
        return s

    def get_extra_params(self):
        return {'n_updates': self.n_updates}

    def set_extra_params(self, x):
        assert set(x.keys()) == {'n_updates'}
        for name, v in x.items():
            setattr(self, name, v)

    def save(self, f):
        pickle.dump(self.get_extra_params(), f, -1)
        pickle.dump([s.get_value(borrow=True) for s in self.optimizer_params],
                    f, -1)

    def load(self, f):
        self.set_extra_params(pickle.load(f))
        values = pickle.load(f)
        if len(values) != len(self.optimizer_params):
            raise ValueError(
                    'Expected %d optimizer parameters, %d in file' % (
                        len(self.optimizer_params), len(values)))
        for s, v in zip(self.optimizer_params, values):
            s.set_value(v)

    def grad_fun(self):
        if self._grad_fun is None:
            self._grad_fun = function(
                    self.inputs + self.outputs,
                    list(self.raw_grad.values()),
                    name='grad_fun')
        return self._grad_fun

    def step(self, *args):
        """Take one optimization step.

        Different subclasses use different rules, but in general this function
        computes gradients and updates `self.params`.

        Parameters
        ----------
        *args : list of numpy.ndarray
            The arguments passed to the step function correspond to the
            arguments in `self.inputs + self.outputs`, i.e. the concatenated
            arrays gives an `inputs` and `outputs` in the constructor.

        Returns
        -------
        loss : float
            Value of the loss function before the current parameter update.
        """
        raise NotImplementedError

    def create_shadows(self, name):
        """Create a set of shared variables of the same shapes as parameters.

        This is useful for creating variables for e.g. gradient squares that
        are used by some of the optimization algorithms.

        Parameters
        ----------
        name : str
            Name prefix to attach to names in the returned object.

        Returns
        -------
        shadows : OrderedDict
            Map of names to shared variables.
        """
        shadows = OrderedDict()
        for param_name, param in self.params.items():
            s = self.shared(np.zeros_like(param.get_value()),
                            name=name+'_'+param_name)
            shadows[name+'_'+param_name] = s
        return shadows


class SGD(Optimizer):
    """Plain Stochastic Gradient Descent (SGD) optimizer.

    To adjust the learning rate, simply modify `self.learning_rate`.

    Parameters
    ----------
    learning_rate : float, optional
        Initial learning rate. Default is 0.01.
    """

    def __init__(self, *args, learning_rate=0.01, **kwargs):
        super().__init__(*args, **kwargs)

        self.learning_rate = learning_rate

        learning_rate = T.scalar('learning_rate')

        updates = [(param, param-(learning_rate*grad))
                   for param, grad
                   in zip(self.params.values(), self.grad.values())]

        self.step1 = function(
                self.inputs+self.outputs+[learning_rate],
                self.loss,
                default_mode=1,
                name='SGD_step1',
                updates=updates)

    def step(self, *args):
        self.n_updates += 1
        return self.step1(*(args + (self.learning_rate,)))


class Nesterov(Optimizer):
    """Nesterov momentum optimizer.

    To adjust the learning rate or momentum parameter, modify
    `self.learning_rate` or `self.momentum`, respectively.

    Implemented as equations (3) and (4) in Sutskever et al. (2013).
    http://jmlr.org/proceedings/papers/v28/sutskever13.pdf

    Parameters
    ----------
    learning_rate : float, optional
        Initial learning rate. Default is 0.01.
    momentum : float, optional
        Initial momentum parameter. Default is 0.9.
    """

    def __init__(self, *args, learning_rate=0.01, momentum=0.9, **kwargs):
        super().__init__(*args, **kwargs)

        self.learning_rate = learning_rate
        self.momentum = momentum

        learning_rate = T.scalar('learning_rate')
        momentum = T.scalar('momentum')

        vs = self.create_shadows('v')

        updates1 = [(p, p + momentum*v)
                    for p,v in zip(self.params.values(), vs.values())]

        updates2 = [(v, momentum*v - learning_rate*grad)
                    for v,grad in zip(vs.values(), self.grad.values())] \
                 + [(p, p - learning_rate*grad)
                    for p,grad in zip(self.params.values(),
                                      self.grad.values())]

        self.step1 = theano.function(
                inputs=[momentum],
                outputs=[],
                name='Nesterov_step1',
                updates=updates1)

        self.step2 = function(
                inputs=self.inputs+self.outputs+[
                    learning_rate, momentum],
                default_mode=1,
                outputs=self.loss,
                name='Nesterov_step2',
                updates=updates2)

    def step(self, *args):
        self.n_updates += 1
        self.step1(self.momentum)
        return self.step2(*(args + (self.learning_rate, self.momentum)))


class RMSProp(Optimizer):
    """RMSProp optimizer.

    Parameters
    ----------
    learning_rate : float, optional
        Initial learning rate.
    decay : float, optional
        Decay rate.
    epsilon : float, optional
        Stabilizing constant.
    """

    def __init__(self, *args, learning_rate=0.001, decay=0.9, epsilon=1e-8,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.learning_rate = learning_rate0
        self.decay = deay0
        self.epsilon = epsilon0

        learning_rate = T.scalar('learning_rate')
        decay = T.scalar('decay')

        squares = self.create_shadows('squares')

        new_squares = [decay*square + (1.0-decay)*T.sqr(g)
                       for g, square in zip(
                           self.grad.values(), squares.values())]

        ds = [-g*learning_rate/T.sqrt(square + self.epsilon)
              for g,square in zip(self.grad.values(), new_squares)]

        updates = [(p, p+d) for p,d in zip(self.params.values(), ds)] \
                + list(zip(squares.values(), new_squares)) \

        self.step1 = function(
                inputs=self.inputs+self.outputs+[
                    learning_rate, decay],
                default_mode=1,
                outputs=self.loss,
                name='RMSProp_step1',
                updates=updates)

    def step(self, *args):
        self.n_updates += 1
        return self.step1(*(args + (self.learning_rate, self.decay)))


class Adam(Optimizer):
    """Adam optimizer.

    To adjust the learning rate, simply modify `self.learning_rate`, although
    the whole point of this algorithm is that you should not need to do this.

    Parameters
    ----------
    learning_rate : float, optional
        Initial learning rate. Default is 0.001.
    beta_1 : float, optional
        First moment decay rate, default: 0.9
    beta_1 : float, optional
        Second moment decay rate, default: 0.999
    epsilon : float, optional
        Stabilizing constant, default: 1e-8

    Kingma and Ba (2014).
    http://arxiv.org/abs/1412.6980
    """

    def __init__(self, *args, learning_rate=0.001, beta_1=0.9,
                 beta_2=0.999, epsilon=1e-8, **kwargs):
        super().__init__(*args, **kwargs)

        self.learning_rate = learning_rate

        learning_rate = T.scalar('learning_rate')

        vs = self.create_shadows('v')
        ms = self.create_shadows('m')
        beta_1_t = self.shared(
                np.asarray(beta_1, dtype=theano.config.floatX),
                name='beta_1_t')
        beta_2_t = self.shared(
                np.asarray(beta_2, dtype=theano.config.floatX),
                name='beta_2_t')

        updates = [(m, beta_1*m + (1.0-beta_1)*g)
                   for m,g in zip(ms.values(), self.grad.values())] \
                + [(v, beta_2*v + (1.0-beta_2)*T.sqr(g))
                   for v,g in zip(vs.values(), self.grad.values())] \
                + [(p, p - (learning_rate*(m/(1.0-beta_1_t))/
                           (T.sqrt(v/(1.0-beta_2_t)) + epsilon)))
                   for p,m,v in zip(self.params.values(),
                                    ms.values(), vs.values())] \
                + [(beta_1_t, beta_1_t * beta_1),
                   (beta_2_t, beta_2_t * beta_2)]

        self.step1 = function(
                inputs=self.inputs+self.outputs+[learning_rate],
                outputs=self.loss,
                default_mode=1,
                name='Adam_step1',
                updates=updates)

    def step(self, *args):
        self.n_updates += 1
        return self.step1(*(args + (self.learning_rate,)))

