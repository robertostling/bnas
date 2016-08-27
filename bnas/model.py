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
from .fun import train_mode


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


class GRU2D(Model):
    """Gated Recurrent Unit using 2D convolutions."""

    def __init__(self, name, side, input_dims, output_dims,
                 filter_dims=3, stride=1,
                 w=None, w_init=None, w_regularizer=None,
                 u=None, u_init=None, u_regularizer=None,
                 b=None, b_init=None, b_regularizer=None):
        super().__init__(name)

        assert output_dims % (side*side) == 0
        assert input_dims % (side*side) == 0
        state_dims = output_dims
        self.side = side
        self.stride = stride
        self.n_in_filters = input_dims // (side*side)
        self.n_filters = output_dims // (side*side)
        self.w_shape = (self.n_filters*3, self.n_in_filters,
                        filter_dims, filter_dims)
        self.u_shape = (self.n_filters*3, self.n_filters,
                        filter_dims, filter_dims)
        self.filter_dims = filter_dims
        self.input_dims = input_dims
        self.state_dims = state_dims

        if w_init is None: w_init = init.Concatenated([
            init.Gaussian(fan_in=filter_dims*filter_dims*self.n_in_filters),
            init.Gaussian(fan_in=filter_dims*filter_dims*self.n_in_filters),
            init.Gaussian(fan_in=filter_dims*filter_dims*self.n_in_filters)],
            axis=0)

        if u_init is None: u_init = init.Concatenated([
            init.Gaussian(fan_in=filter_dims*filter_dims*self.n_filters),
            init.Gaussian(fan_in=filter_dims*filter_dims*self.n_filters),
            init.Gaussian(fan_in=filter_dims*filter_dims*self.n_filters)],
            axis=0)

        if b_init is None: b_init = init.Concatenated([
            init.Constant(0.0),
            init.Constant(0.0),
            init.Constant(0.0)])

        self.param('w', (self.n_filters*3, self.n_in_filters,
                         filter_dims, filter_dims),
                   init_f=w_init, value=w)
        self.param('u', (self.n_filters*3, self.n_filters,
                         filter_dims, filter_dims),
                   init_f=u_init, value=u)
        self.param('b', (self.n_filters*3,), init_f=b_init, value=b)

        self.regularize(self._w, w_regularizer)
        self.regularize(self._u, u_regularizer)
        self.regularize(self._b, b_regularizer)

    def __call__(self, inputs, state):
        batch_size = inputs.shape[0]
        state2d = state.reshape(
                (batch_size, self.n_filters, self.side, self.side))
        inputs2d = inputs.reshape(
                (batch_size, self.n_in_filters, self.side, self.side))
        x_zrh = T.nnet.conv2d(
                inputs2d,
                self._w,
                input_shape=(None, self.n_in_filters, self.side, self.side),
                filter_shape=(self.n_filters*3, self.n_in_filters,
                              self.filter_dims, self.filter_dims),
                border_mode='half',
                subsample=(self.stride, self.stride),
                filter_flip=True) + self._b.dimshuffle('x',0,'x','x')
        u_zr = T.nnet.conv2d(
                state2d,
                self._u[:self.n_filters*2,:,:,:],
                input_shape=(None, self.n_filters, self.side, self.side),
                filter_shape=(self.n_filters*2, self.n_filters,
                              self.filter_dims, self.filter_dims),
                border_mode='half',
                subsample=(self.stride, self.stride),
                filter_flip=True)
        zr = T.nnet.sigmoid(x_zrh[:,:self.n_filters*2,:,:] + u_zr)
        z = zr[:,:self.n_filters,:,:]
        r = zr[:,self.n_filters:,:,:]
        x_h = x_zrh[:,self.n_filters*2:,:,:]
        u_h = T.nnet.conv2d(
                r * state2d,
                self._u[self.n_filters*2:,:,:,:],
                input_shape=(None, self.n_filters, self.side, self.side),
                filter_shape=(self.n_filters, self.n_filters,
                              self.filter_dims, self.filter_dims),
                border_mode='half',
                subsample=(self.stride, self.stride),
                filter_flip=True)
        hh = T.tanh(x_h + u_h)
        h = z*state2d + (1.0-z)*hh
        return h.reshape(state.shape)


class StackedRNN(Model):
    """Stacked recurrent unit of a specified type.

    Parameters
    ----------
    name : str
        Name of layer.
    gate : class
        Recurrent gate class, e.g. :class:`.GRU` or :class:`.IRNN`
    input_dims : int
        Number of dimensions in input (passed directly to `gate`)
    state_dims : int
        The product of `n_layers` and the hidden state dimensionality of each
        layer. For instance, if you stack 4 GRUs, each with an `output_dims`
        of 1000, this argument should be 4000.
    n_layers : int
        Number of layers.
    """

    def __init__(self, name, gate, input_dims, state_dims, n_layers, **kwargs):
        super().__init__(name)

        if state_dims % n_layers != 0:
            raise ValueError(
                'A stacked RNN must have a state dimensionality which is a '
                'multiple of the number of layers, but state_dims = %d and '
                'n_layers = %d' % (state_dims, n_layers))

        self.n_layers = n_layers
        output_dims = state_dims // n_layers
        self.input_dims = input_dims
        self.output_dims = output_dims

        self.add(gate('layer0', input_dims, output_dims, **kwargs))
        for i in range(1, n_layers):
            # TODO: should use init.Orthogonal() here
            self.add(gate('layer%d' % i, output_dims, output_dims, **kwargs))

    def __call__(self, inputs, state):
        if self.n_layers == 1: return self.layer0(inputs, state)
        outputs = [self.layer0(inputs, state[:, :self.output_dims])]
        for i in range(1, self.n_layers):
            outputs.append(getattr(self, 'layer%d' % i)(
                outputs[-1],
                state[:, i*self.output_dims:(i+1)*self.output_dims]))
        return T.concatenate(outputs, axis=-1)


class RHN(Model):
    """Recurrent Highway Network."""

    def __init__(self, name, input_dims, output_dims,
                 depth=4,
                 w=None, w_init=None, w_regularizer=None,
                 us=None, u_init=None, u_regularizer=None,
                 b=None, b_init=None, b_regularizer=None,
                 use_layernorm=False):
        super().__init__(name)

        state_dims = output_dims

        self.depth = depth
        self.state_dims = state_dims
        self.use_layernorm = use_layernorm

        if w_init is None: w_init = init.Concatenated([
            init.Gaussian(fan_in=input_dims),
            init.Gaussian(fan_in=input_dims)],
            axis=1)

        if u_init is None: u_init = init.Concatenated([
            init.Orthogonal(),
            init.Orthogonal()],
            axis=1)

        if b_init is None: b_init = init.Concatenated([
            init.Constant(0.0),
            init.Constant(1.0)])

        self.param('w', (input_dims, state_dims*2), init_f=w_init, value=w)
        for i in range(depth):
            u = None if us is None else us[i]
            self.param('u%d'%i, (state_dims, state_dims*2),
                       init_f=u_init, value=u)
        self.param('b', (state_dims*2,), init_f=b_init, value=b)

        self.regularize(self._w, w_regularizer)
        for i in range(depth):
            self.regularize(getattr(self, '_u%d'%i), u_regularizer)
        self.regularize(self._b, b_regularizer)

        if use_layernorm:
            for i in range(depth):
                self.add(LayerNormalization('ln_h%d'%i, (None, state_dims)))

    def __call__(self, inputs, state):
        s = state
        for i in range(self.depth):
            if self.use_layernorm:
                s = getattr(self, 'ln_h%d'%i)(s)
            ht = T.dot(s, getattr(self, '_u%d'%i)) + self._b
            if i == 0:
                ht = ht + T.dot(inputs, self._w)
            h = T.tanh(ht[:,:self.state_dims])
            t = T.nnet.sigmoid(ht[:,self.state_dims:])
            c = 1.0 - t
            s = s*t + h*c
        return s


class LSTM(Model):
    """Long Short-Term Memory.

    Note that both states are merged in the same vector, so output_dims must
    be even.
    """

    def __init__(self, name, input_dims, output_dims,
                 w=None, w_init=None, w_regularizer=None,
                 u=None, u_init=None, u_regularizer=None,
                 b=None, b_init=None, b_regularizer=None,
                 dropout=0,
                 use_layernorm=False):
        super().__init__(name)

        if output_dims % 2 != 0:
            raise ValueError(
                    'LSTM output_dims must be even, is %d' % output_dims)
        state_dims = output_dims // 2
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.state_dims = state_dims
        self.use_layernorm = use_layernorm
        self.dropout = dropout

        if self.dropout:
            self.rng = RandomStreams()

        if w_init is None: w_init = init.Concatenated([
            init.Gaussian(fan_in=input_dims),
            init.Gaussian(fan_in=input_dims),
            init.Gaussian(fan_in=input_dims),
            init.Gaussian(fan_in=input_dims)],
            axis=1)

        if u_init is None: u_init = init.Concatenated([
            init.Orthogonal(),
            init.Orthogonal(),
            init.Orthogonal(),
            init.Orthogonal()],
            axis=1)

        if b_init is None: b_init = init.Concatenated([
            init.Constant(0.0),
            init.Constant(1.0),
            init.Constant(0.0),
            init.Constant(0.0)])

        self.param('w', (input_dims, state_dims*4), init_f=w_init, value=w)
        self.param('u', (state_dims, state_dims*4), init_f=u_init, value=u)
        self.param('b', (state_dims*4,), init_f=b_init, value=b)

        self.regularize(self._w, w_regularizer)
        self.regularize(self._u, u_regularizer)
        self.regularize(self._b, b_regularizer)

        if use_layernorm == 'full':
            self.add(LayerNormalization('ln_i', (None, state_dims)))
            self.add(LayerNormalization('ln_f', (None, state_dims)))
            self.add(LayerNormalization('ln_o', (None, state_dims)))
            self.add(LayerNormalization('ln_c', (None, state_dims)))
        if use_layernorm:
            self.add(LayerNormalization('ln_h', (None, state_dims)))

    def dropout_masks(self, *shapes):
        if self.dropout:
            p = 1.0-self.dropout
            return [
                self.rng.binomial(shape, p=p).astype(theano.config.floatX) / p
                for shape in shapes]
        else:
            return []

    def __call__(self, inputs, state,
                 input_dropout_mask=None, h_dropout_mask=None,
                 *non_sequences):
        h_tm1 = state[:, :self.state_dims]
        c_tm1 = state[:, self.state_dims:]
        if self.dropout:
            if not input_dropout_mask is None:
                inputs = ifelse(
                        train_mode,
                        inputs * input_dropout_mask,
                        inputs)
            if not h_dropout_mask is None:
                h_tm1 = ifelse(
                        train_mode,
                        h_tm1 * h_dropout_mask,
                        h_tm1)
        x = T.dot(inputs, self._w) + T.dot(h_tm1, self._u)
        x = x + self._b.dimshuffle('x', 0)
        if self.use_layernorm == 'full':
            i = T.nnet.sigmoid(self.ln_i(
                x[:,0*self.state_dims:1*self.state_dims]))
            f = T.nnet.sigmoid(self.ln_f(
                x[:,1*self.state_dims:2*self.state_dims]))
            o = T.nnet.sigmoid(self.ln_o(
                x[:,2*self.state_dims:3*self.state_dims]))
            c = T.tanh(        self.ln_c(
                x[:,3*self.state_dims:4*self.state_dims]))
        else:
            i = T.nnet.sigmoid(x[:,0*self.state_dims:1*self.state_dims])
            f = T.nnet.sigmoid(x[:,1*self.state_dims:2*self.state_dims])
            o = T.nnet.sigmoid(x[:,2*self.state_dims:3*self.state_dims])
            c = T.tanh(        x[:,3*self.state_dims:4*self.state_dims])
        c_t = f*c_tm1 + i*c
        h_t = o*T.tanh(self.ln_h(c_t) if self.use_layernorm else c_t)
        state = T.concatenate([h_t, c_t], axis=-1)
        return state


class GRU(Model):
    """Gated Recurrent Unit."""

    def __init__(self, name, input_dims, output_dims,
                 w=None, w_init=None, w_regularizer=None,
                 u=None, u_init=None, u_regularizer=None,
                 b=None, b_init=None, b_regularizer=None,
                 use_layernorm=False):
        super().__init__(name)

        state_dims = output_dims
        self.input_dims = input_dims
        self.state_dims = state_dims
        self.use_layernorm = use_layernorm

        if w_init is None: w_init = init.Concatenated([
            init.Gaussian(fan_in=input_dims),
            init.Gaussian(fan_in=input_dims),
            init.Gaussian(fan_in=input_dims)],
            axis=1)

        if u_init is None: u_init = init.Concatenated([
            init.Orthogonal(),
            init.Orthogonal(),
            init.Orthogonal()],
            axis=1)

        if b_init is None: b_init = init.Constant(0.0)

        self.param('w', (input_dims, state_dims*3), init_f=w_init, value=w)
        self.param('u', (state_dims, state_dims*3), init_f=u_init, value=u)
        self.param('b', (state_dims*3,), init_f=b_init, value=b)

        self.regularize(self._w, w_regularizer)
        self.regularize(self._u, u_regularizer)
        self.regularize(self._b, b_regularizer)

        # TODO: the layernorm application does not seem very good compared to
        # the LSTM case, at least not for char LM. Try other variants.
        if use_layernorm:
            self.add(LayerNormalization('ln_h', (None, state_dims*1)))
        if use_layernorm == 'full':
            self.add(LayerNormalization('ln_h_z', (None, state_dims*1)))
            self.add(LayerNormalization('ln_h_r', (None, state_dims*1)))
            self.add(LayerNormalization('ln_h_h', (None, state_dims*1)))

    def __call__(self, inputs, state):
        if self.use_layernorm:
            state = self.ln_h(state)
        b = self._b.dimshuffle('x', 0)
        x_zrh = T.dot(inputs, self._w)
        h_zrh = T.dot(state, self._u)
        x_z = x_zrh[:, :self.state_dims]
        h_z = h_zrh[:, :self.state_dims]
        x_r = x_zrh[:, self.state_dims:self.state_dims*2]
        h_r = h_zrh[:, self.state_dims:self.state_dims*2]
        x_h = x_zrh[:, self.state_dims*2:]
        h_h = h_zrh[:, self.state_dims*2:]
        if self.use_layernorm == 'full':
           h_z = self.ln_h_z(h_z)
           h_r = self.ln_h_r(h_r)
           h_h = self.ln_h_h(h_h)
        z = T.nnet.sigmoid(x_z + h_z + b[:, :self.state_dims])
        r = T.nnet.sigmoid(x_r + h_r + b[:, self.state_dims:self.state_dims*2])
        hh = T.tanh(x_h + r*h_h + b[:, self.state_dims*2:]) 
        h = (1.0-z)*state + z*hh
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


class Dropout(Model):
    """Dropout layer."""

    def __init__(self, name, dropout):
        super().__init__(name)
        self.p = 1.0 - dropout
        self.rng = RandomStreams()

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

