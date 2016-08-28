"""Network models and submodels.

The :class:`Model` class is used to encapsulate a set of Theano shared
variables (model parameters), and can create symbolic expressions for model
outputs and loss functions.

This module also contains subclasses, such as :class:`Linear`, that function
as building blocks for more complex networks.
"""

from collections import OrderedDict
import pickle

import numpy as np
import theano
from theano.ifelse import ifelse
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano import tensor as T

from . import init
from .fun import train_mode, function


class Model:
    """Base class for neural network models.

    Attributes
    ----------
    name : str
        Name of the model.
    params : OrderedDict of str -> :class:`theano.compile.sharedvalue.SharedVariable`
        Mapping from parameter names to Theano shared variables. Note that
        submodel parameters are not included, so this should normally not be
        accessed directly, rather use `self.parameters()`.
    regularization : list of Theano symbolic expressions
        These expressions should all be added to the loss function when
        optimizing. Use `self.regularize()` to modify.
    """

    def __init__(self, name):
        """Initialize an empty model.

        Parameters
        ----------
        name : str
            Name of the model.
        """
        self.name = name
        self.params = OrderedDict()
        self.regularization = []
        self.submodels = OrderedDict()

    def loss(self):
        """Part of the loss function that is independent of inputs."""
        terms = [submodel.loss() for submodel in self.submodels.values()] \
              + self.regularization
        return sum(terms, T.as_tensor_variable(0.0))

    def parameters(self, include_submodels=True):
        """Iterate over the parameters of this model and its submodels.
        
        Each value produced by the iterator is a tuple (name, value), where
        the name is a tuple of strings describing the hierarchy of submodels,
        e.g. ('hidden', 'b'), and the value is a Theano shared variable.

        Parameters
        ----------
        include_submodels : bool
            If ``True`` (default), also iterate over submodel parameters.
        """
        for name, p in self.params.items():
            yield ((name,), p)
        if include_submodels:
            for submodel in self.submodels.values():
                for name, p in submodel.parameters():
                    yield ((submodel.name,) + name, p)

    def parameters_list(self, include_submodels=True):
        """Return a list with parameters, without their names."""
        return list(p for name, p in
                self.parameters(include_submodels=include_submodels))

    def parameter(self, name):
        """Return the parameter with the given name.
        
        Parameters
        ----------
        name : tuple of str
            Path to variable, e.g. ('hidden', 'b') to find the parameter 'b'
            in the submodel 'hidden'.
        
        Returns
        -------
        value : :class:`theano.compile.sharedvalue.SharedVariable`
        """

        if not isinstance(name, tuple):
            raise TypeError('Expected tuple, got %s' % type(name))
        if len(name) == 1:
            return self.param[name]
        elif len(name) >= 2:
            return self.submodels[name[0]].parameter(name[1:])
        else:
            raise ValueError('Name tuple must not be empty!')

    def param(self, name, dims, init_f=None,
              value=None, dtype=theano.config.floatX):
        """Create a new parameter, or share an existing one.

        Parameters
        ----------
        name : str
            Name of parameter, this will be used directly in `self.params`
            and used to create `self._name`.
        dims : tuple
            Shape of the parameter vector.
        value : :class:`theano.compile.sharedvalue.SharedVariable`, optional
            If this parameter should be shared, a SharedVariable instance can
            be passed here.
        init_f : (tuple => numpy.ndarray)
            Function used to initialize the parameter vector.
        dtype : str or numpy.dtype
            Data type (default is `theano.config.floatX`)

        Returns
        -------
        p : :class:`theano.compile.sharedvalue.SharedVariable`
        """
        if name in self.params:
            if not value is None:
                raise ValueError('Trying to add a shared parameter (%s), '
                                 'but a parameter with the same name already '
                                 'exists in %s!' % (name, self.name))
            return self.params[name]
        if value is None:
            if init_f is None:
                raise ValueError('Creating new parameter, but no '
                                 'initialization specified!')
            p = theano.shared(init_f(dims, dtype=dtype), name=name)
            self.params[name] = p
        else:
            p = value
        setattr(self, '_'+name, p)
        return p

    def regularize(self, p, regularizer):
        """Add regularization to a parameter.

        Parameters
        ----------
        p : :class:`theano.compile.sharedvalue.SharedVariable`
            Parameter to apply regularization
        regularizer : function
            Regularization function, which should return a symbolic
            expression.
        """
        if not regularizer is None:
            self.regularization.append(regularizer(p))

    def add(self, submodel):
        """Import parameters from a submodel.
        
        If a submodel named "hidden" has a parameter "b", it will be imported
        as "hidden_b", also accessible as `self._hidden_b`.

        Parameters
        ----------
        submodel : :class:`.Model`

        Returns
        -------
        submodel : :class:`.Model`
            Equal to the parameter, for convenience.
        """
        if submodel.name in self.submodels:
            raise ValueError('Submodel with name %s already exists in %s!' % (
                submodel.name, self.name))
        self.submodels[submodel.name] = submodel
        setattr(self, submodel.name, submodel)
        return submodel

    def save(self, f, include_submodels=True):
        """Save the parameter values of this model to a file object.

        Parameters
        ----------
        f : file
            File object to write to, assumed to be opened in 'wb' mode.
        include_submodels : bool
            If ``True`` (default), also save submodel parameters.
        """
        pickle.dump({name: p.get_value(borrow=True)
                     for name, p in self.parameters(
                         include_submodels=include_submodels)},
                    f, -1)

    def load(self, f, allow_incomplete=False, allow_unused=False):
        """Load (some) weights of this model from a file object.
        
        Parameters
        ----------
        f : file
            File object to read from, assumeb to be opened in 'rb' mode.
        allow_incomplete : bool
            If ``False``, throw a `ValueError` if some model parameters are
            missing in the file.
        allow_unused : bool
            If ``False``, throw a `ValueError` if the file contains model
            parameters that are not used in this model.
        """
        data = pickle.load(f)
        parameters = dict(self.parameters())
        names = frozenset(data.keys()) & frozenset(parameters.keys())
        if not allow_incomplete and len(names) < len(parameters):
            raise ValueError(
                    'The following parameters are missing: %s' % ', '.join(
                        sorted(frozenset(parameters.keys()) - names)))
        if not allow_unused and len(names) < len(data):
            raise ValueError(
                    'The following parameters are unused: %s' % ', '.join(
                        sorted(frozenset(data.keys()) - names)))
        for name in names:
            value = data[name]
            old_value = parameters[name].get_value(borrow=True)
            if value.shape != old_value.shape:
                raise ValueError(
                        'Loaded shape is %s but %s expected' % (
                            value.shape, old_value.shape))
            parameters[name].set_value(value)

    def compile(self, *args):
        return function(list(args), self(*args))


class Linear(Model):
    """Fully connected linear layer.

    This layer creates one shared parameter, `name_w` of shape
    `(input_dims, output_dims)` if `use_bias` is ``False``, otherwise it
    also creates `name_b` of shape `output_dims` for biases.

    Parameters
    ----------
    name : str
        Name of layer.
    input_dims : int
        Number of inputs.
    output_dims : int
        Number of outputs.
    w : :class:`theano.compile.sharedvalue.SharedVariable`
        Weight vector to use, or pass ``None`` (default) to create a new
        one.
    w_init : :class:`.init.InitializationFunction`
        Initialization for weight vector, in case `w` is ``None``.
    w_regularizer : :class:`.regularize.Regularizer`, optional
        Regularization for weight matrix.
    b : :class:`theano.compile.sharedvalue.SharedVariable`
        Bias vector to use, or pass ``None`` (default) to create a new
        one.
    b_init : :class:`.init.InitializationFunction`
        Initialization for bias vector, in case `b` is ``None``.
    b_regularizer : :class:`.regularize.Regularizer`, optional
        Regularization for biases.
    use_bias : bool
        If ``False``, no bias is used and the `b` and `b_init` parameters
        are ignored.
    """
    def __init__(self, name, input_dims, output_dims,
                 w=None, w_init=None, w_regularizer=None,
                 b=None, b_init=None, b_regularizer=None,
                 use_bias=True):
        super().__init__(name)

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.use_bias = use_bias

        if w_init is None: w_init = init.Gaussian(fan_in=input_dims)
        if b_init is None: b_init = init.Constant(0.0)

        self.param('w', (input_dims, output_dims), init_f=w_init, value=w)
        self.regularize(self._w, w_regularizer)
        if use_bias:
            self.param('b', (output_dims,), init_f=b_init, value=b)
            self.regularize(self._b, b_regularizer)

    def __call__(self, inputs):
        if self.use_bias:
            return T.dot(inputs, self._w) + self._b
        else:
            return T.dot(inputs, self._w)


class Conv1D(Model):
    """1D convolution layer with linear activations.

    The input shape is assumed to be (batch_size, length, dims).
    """

    def __init__(self, name, input_dims, output_dims,
                 filter_dims=3, stride=1,
                 f=None, f_init=None, f_regularizer=None,
                 b=None, b_init=None, b_regularizer=None):
        super().__init__(name)
        if f_init is None:
            f_init = init.Gaussian(fan_in=filter_dims*input_dims)
        if b_init is None:
            b_init = init.Constant(0.0)
        self.stride = stride
        self.input_dims = input_dims
        self.f_shape = (output_dims, input_dims, filter_dims, 1)
        self.param('f', self.f_shape, init_f=f_init)
        self.param('b', (output_dims,), init_f=b_init)

    def __call__(self, inputs, inputs_mask):
        x = T.nnet.conv2d(
                (inputs * inputs_mask.dimshuffle(0,1,'x')
                    ).dimshuffle(0,2,1,'x'),
                self._f,
                input_shape=(None, self.input_dims, None, 1),
                filter_shape=self.f_shape,
                border_mode='half',
                subsample=(self.stride, 1),
                filter_flip=True)

        batch_size = inputs.shape[0]
        length = inputs.shape[1]
        dims = inputs.shape[2]

        x = x.reshape((batch_size, dims, length)).dimshuffle(0,2,1)

        return x + self._b.dimshuffle('x','x',0)


class LSTM(Model):
    """Long Short-Term Memory."""

    def __init__(self, name, input_dims, state_dims,
                 w=None, w_init=None, w_regularizer=None,
                 u=None, u_init=None, u_regularizer=None,
                 b=None, b_init=None, b_regularizer=None,
                 layernorm=False):
        super().__init__(name)

        assert layernorm in (False, 'c', 'all')

        self.input_dims = input_dims
        self.state_dims = state_dims
        self.layernorm = layernorm

        if w_init is None: w_init = init.Concatenated(
            [init.Gaussian(fan_in=input_dims)] * 4, axis=1)

        if u_init is None: u_init = init.Concatenated(
            [init.Orthogonal()]*4, axis=1)

        if b_init is None: b_init = init.Concatenated(
            [init.Constant(x) for x in [0.0, 1.0, 0.0, 0.0]])

        self.param('w', (input_dims, state_dims*4), init_f=w_init, value=w)
        self.param('u', (state_dims, state_dims*4), init_f=u_init, value=u)
        self.param('b', (state_dims*4,), init_f=b_init, value=b)

        self.regularize(self._w, w_regularizer)
        self.regularize(self._u, u_regularizer)
        self.regularize(self._b, b_regularizer)

        if layernorm == 'all':
            self.add(LayerNormalization('ln_i', (None, state_dims)))
            self.add(LayerNormalization('ln_f', (None, state_dims)))
            self.add(LayerNormalization('ln_o', (None, state_dims)))
            self.add(LayerNormalization('ln_c', (None, state_dims)))
        if layernorm:
            self.add(LayerNormalization('ln_h', (None, state_dims)))

    def __call__(self, inputs, h_tm1, c_tm1):
        x = T.dot(inputs, self._w) + T.dot(h_tm1, self._u)
        x = x + self._b.dimshuffle('x', 0)
        def x_part(i): return x[:, i*self.state_dims:(i+1)*self.state_dims]
        if self.layernorm == 'all':
            i = T.nnet.sigmoid(self.ln_i(x_part(0)))
            f = T.nnet.sigmoid(self.ln_f(x_part(1)))
            o = T.nnet.sigmoid(self.ln_o(x_part(2)))
            c = T.tanh(        self.ln_c(x_part(3)))
        else:
            i = T.nnet.sigmoid(x_part(0))
            f = T.nnet.sigmoid(x_part(1))
            o = T.nnet.sigmoid(x_part(2))
            c = T.tanh(        x_part(3))
        c_t = f*c_tm1 + i*c
        h_t = o*T.tanh(self.ln_h(c_t) if self.layernorm else c_t)
        return h_t, c_t


class Dropout(Model):
    """Dropout layer."""

    def __init__(self, name, dropout):
        super().__init__(name)
        self.p = 1.0 - dropout
        self.rng = RandomStreams()

    def mask(self, shape):
        """Return a scaled mask for a (symbolic) shape.

        This can be used for dropout in recurrent layers, where a fixed mask
        is passed through the non_sequences argument to theano.scan().
        """
        m = self.rng.binomial(shape, p=self.p).astype(theano.config.floatX)
        return m / self.p

    def __call__(self, inputs):
        return ifelse(
                train_mode,
                inputs * (self.rng.binomial(inputs.shape, p=self.p).astype(
                    theano.config.floatX) / self.p),
                inputs)


class LayerNormalization(Model):
    """Layer Normalization (Ba, Kiros and Hinton 2016)."""

    def __init__(self, name, inputs_shape, g_init=None, axis=-1, epsilon=1e-6):
        super().__init__(name)

        self.inputs_shape = inputs_shape
        self.axis = axis
        self.epsilon = epsilon
        if g_init is None: g_init = init.Constant(1.0)

        self.param('g', (inputs_shape[self.axis],), init_f=g_init)

    def __call__(self, inputs):
        broadcast = ['x']*len(self.inputs_shape)
        broadcast[self.axis] = 0

        mean = inputs.mean(axis=self.axis, keepdims=True)
        std = inputs.std(axis=self.axis, keepdims=True)
        normed = (inputs - mean) / (std + self.epsilon)
        return normed * self._g.dimshuffle(*broadcast)

