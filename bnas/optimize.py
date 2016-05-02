from collections import OrderedDict

import numpy as np
import theano
from theano import tensor as T


class Optimizer:
    def __init__(self, params, loss, inputs=[], outputs=[], grad_max_norm=5.0):
        self.params = params
        self.loss = loss
        self.inputs = inputs
        self.outputs = outputs
        self.grad_max_norm = grad_max_norm

        self.grad = OrderedDict((name, T.grad(loss, param))
                                for name, param in params.items())


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
            s = theano.shared(np.zeros_like(param.get_value()))
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.learning_rate = kwargs.get('learning_rate', 0.01)

        learning_rate = T.scalar('learning_rate')

        updates = [(param, param-(learning_rate*grad))
                   for param, grad
                   in zip(self.params.values(), self.grad.values())]

        self.step1 = theano.function(
                self.inputs+self.outputs+[learning_rate],
                self.loss,
                updates=updates)

    def step(self, *args):
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.learning_rate = kwargs.get('learning_rate', 0.01)
        self.momentum = kwargs.get('momentum', 0.9)

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
                updates=updates1)

        self.step2 = theano.function(
                inputs=self.inputs+self.outputs+[learning_rate, momentum],
                outputs=self.loss,
                updates=updates2)

    def step(self, *args):
        self.step1(self.momentum)
        return self.step2(*(args + (self.learning_rate, self.momentum)))


class Adam(Optimizer):
    """Adam optimizer.

    To adjust the learning rate, simply modify `self.learning_rate`, although
    the whole point of this algorithm is that you should not need to do this.

    Parameters
    ----------
    learning_rate : float, optional
        Initial learning rate. Default is 0.01.
    beta_1 : float, optional
        First moment decay rate, default: 0.9
    beta_1 : float, optional
        Second moment decay rate, default: 0.999
    epsilon : float, optional
        Stabilizing constant, default: 1e-8

    Kingma and Ba (2014).
    http://arxiv.org/abs/1412.6980
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.learning_rate = kwargs.get('learning_rate', 0.001)
        beta_1 = kwargs.get('beta_1', 0.9)
        beta_2 = kwargs.get('beta_2', 0.999)
        epsilon = kwargs.get('epsilon', 1e-8)

        learning_rate = T.scalar('learning_rate')

        vs = self.create_shadows('v')
        ms = self.create_shadows('m')
        beta_1_t = theano.shared(
                np.asarray(beta_1, dtype=theano.config.floatX),
                name='beta_1_t')
        beta_2_t = theano.shared(
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

        self.step1 = theano.function(
                inputs=self.inputs+self.outputs+[learning_rate],
                outputs=self.loss,
                updates=updates)

    def step(self, *args):
        return self.step1(*(args + (self.learning_rate,)))

