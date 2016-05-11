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
from theano import tensor as T

from . import init


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


class GRU(Model):
    """Gated Recurrent Unit."""

    def __init__(self, name, input_dims, output_dims,
                 w_z=None, w_z_init=None, w_z_regularizer=None,
                 w_r=None, w_r_init=None, w_r_regularizer=None,
                 w_h=None, w_h_init=None, w_h_regularizer=None,
                 u_z=None, u_z_init=None, u_z_regularizer=None,
                 u_r=None, u_r_init=None, u_r_regularizer=None,
                 u_h=None, u_h_init=None, u_h_regularizer=None,
                 b_z=None, b_z_init=None, b_z_regularizer=None,
                 b_r=None, b_r_init=None, b_r_regularizer=None,
                 b_h=None, b_h_init=None, b_h_regularizer=None,
                 use_bias=True):
        super().__init__(name)

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.use_bias = use_bias

        if w_z_init is None: w_z_init = init.Gaussian(fan_in=input_dims)
        if w_r_init is None: w_r_init = init.Gaussian(fan_in=input_dims)
        if w_h_init is None: w_h_init = init.Gaussian(fan_in=input_dims)
        if u_z_init is None: u_z_init = init.Orthogonal()
        if u_r_init is None: u_r_init = init.Orthogonal()
        if u_h_init is None: u_h_init = init.Orthogonal()
        if self.use_bias:
            if b_z_init is None: b_z_init = init.Constant(0.0)
            if b_r_init is None: b_r_init = init.Constant(0.0)
            if b_h_init is None: b_h_init = init.Constant(0.0)
        
        self.param('w_z',(input_dims,output_dims),  init_f=w_z_init, value=w_z)
        self.param('w_r',(input_dims,output_dims),  init_f=w_r_init, value=w_r)
        self.param('w_h',(input_dims,output_dims),  init_f=w_h_init, value=w_h)
        self.param('u_z',(output_dims,output_dims), init_f=u_z_init, value=u_z)
        self.param('u_r',(output_dims,output_dims), init_f=u_r_init, value=u_r)
        self.param('u_h',(output_dims,output_dims), init_f=u_h_init, value=u_h)
        if self.use_bias:
            self.param('b_z',(output_dims,), init_f=b_z_init, value=b_z)
            self.param('b_r',(output_dims,), init_f=b_r_init, value=b_r)
            self.param('b_h',(output_dims,), init_f=b_h_init, value=b_h)

        self.regularize(self._w_z, w_z_regularizer)
        self.regularize(self._w_r, w_r_regularizer)
        self.regularize(self._w_h, w_h_regularizer)
        self.regularize(self._u_z, u_z_regularizer)
        self.regularize(self._u_r, u_r_regularizer)
        self.regularize(self._u_h, u_h_regularizer)
        if self.use_bias:
            self.regularize(self._b_z, b_z_regularizer)
            self.regularize(self._b_r, b_r_regularizer)
            self.regularize(self._b_h, b_h_regularizer)

    def __call__(self, inputs, state):
        if self.use_bias:
            x_z = T.dot(inputs, self._w_z) + self._b_z
            x_r = T.dot(inputs, self._w_r) + self._b_r
            x_h = T.dot(inputs, self._w_h) + self._b_h
        else:
            x_z = T.dot(inputs, self._w_z)
            x_r = T.dot(inputs, self._w_r)
            x_h = T.dot(inputs, self._w_h)
        z = T.nnet.sigmoid(x_z + T.dot(state, self._u_z))
        r = T.nnet.sigmoid(x_r + T.dot(state, self._u_r))
        hh = T.tanh(x_h + T.dot(r * state, self._u_h))
        h = z*state + (1-z)*hh
        return h


class DSGU(Model):
    """Deep Simple Gated Unit.
    
    http://arxiv.org/pdf/1604.02910v2.pdf
    """

    def __init__(self, name, input_dims, output_dims,
                 w_xh=None, w_xh_init=None, w_xh_regularizer=None,
                 w_zxh=None, w_zxh_init=None, w_zxh_regularizer=None,
                 w_go=None, w_go_init=None, w_go_regularizer=None,
                 w_xz=None, w_xz_init=None, w_xz_regularizer=None,
                 w_hz=None, w_hz_init=None, w_hz_regularizer=None,
                 b_g=None, b_g_init=None, b_g_regularizer=None,
                 b_z=None, b_z_init=None, b_z_regularizer=None,
                 use_bias=True):
        super().__init__(name)

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.use_bias = use_bias

        if w_xh_init is None: w_xh_init = init.Gaussian(fan_in=input_dims)
        if w_xz_init is None: w_xz_init = init.Gaussian(fan_in=input_dims)
        if w_zxh_init is None: w_zxh_init = init.Orthogonal()
        if w_go_init is None: w_go_init = init.Orthogonal()
        if w_hz_init is None: w_hz_init = init.Orthogonal()
        if self.use_bias:
            if b_z_init is None: b_z_init = init.Constant(0.0)
            if b_g_init is None: b_g_init = init.Constant(0.0)
        
        self.param('w_xh', (input_dims,output_dims),
                   init_f=w_xh_init, value=w_xh)
        self.param('w_xz', (input_dims,output_dims),
                   init_f=w_xz_init, value=w_xz)
        self.param('w_zxh', (output_dims,output_dims),
                   init_f=w_zxh_init, value=w_zxh)
        self.param('w_go', (output_dims,output_dims),
                   init_f=w_go_init, value=w_go)
        self.param('w_hz', (output_dims,output_dims),
                   init_f=w_hz_init, value=w_hz)

        if self.use_bias:
            self.param('b_z',(output_dims,), init_f=b_z_init, value=b_z)
            self.param('b_g',(output_dims,), init_f=b_g_init, value=b_g)

        self.regularize(self._w_xh,  w_xh_regularizer)
        self.regularize(self._w_zxh, w_zxh_regularizer)
        self.regularize(self._w_go,  w_go_regularizer)
        self.regularize(self._w_xz,  w_xz_regularizer)
        self.regularize(self._w_hz,  w_hz_regularizer)
        if self.use_bias:
            self.regularize(self._b_z, b_z_regularizer)
            self.regularize(self._b_g, b_g_regularizer)

    def __call__(self, inputs, state):
        if self.use_bias:
            x_g = T.dot(inputs, self._w_xh) + self._b_g
        else:
            x_g = T.dot(inputs, self._w_xh)
        z_g = T.tanh(T.dot(x_g * state, self._w_zxh))
        z_o = T.nnet.sigmoid(T.dot(z_g * state, self._w_go))
        if self.use_bias:
            z_t = T.nnet.hard_sigmoid(
                      T.dot(inputs, self._w_xz)
                    + T.dot(state, self._w_hz)
                    + self._b_z)
        else:
            z_t = T.nnet.hard_sigmoid(
                      T.dot(inputs, self._w_xz)
                    + T.dot(state, self._w_hz))
        return (1.0-z_t)*state + z_t*z_o


class IRNN(Model):
    """Identity Recurrent Neural Network (iRNN)."""

    def __init__(self, name, input_dims, output_dims,
                 w=None, w_init=None, w_regularizer=None,
                 u=None, u_init=None, u_regularizer=None,
                 b=None, b_init=None, b_regularizer=None,
                 use_bias=True):
        super().__init__(name)

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.use_bias = use_bias

        if w_init is None: w_init = init.Gaussian(fan_in=input_dims)
        if u_init is None: u_init = init.Identity()
        if self.use_bias:
            if b_init is None: b_init = init.Constant(0.0)

        self.param('w',(input_dims,output_dims), init_f=w_init, value=w)
        self.param('u',(output_dims,output_dims), init_f=u_init, value=u)
        if self.use_bias:
            self.param('b',(output_dims,), init_f=b_init, value=b)

        self.regularize(self._w, w_regularizer)
        self.regularize(self._u, u_regularizer)
        if self.use_bias:
            self.regularize(self._b, b_regularizer)

    def __call__(self, inputs, state):
        h = T.dot(inputs, self._w) + T.dot(state, self._u)
        return T.nnet.relu(h + self._b) if self.use_bias else T.nnet.relu(h)

