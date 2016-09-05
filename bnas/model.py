"""Network models and submodels.

The :class:`Model` class is used to encapsulate a set of Theano shared
variables (model parameters), and can create symbolic expressions for model
outputs and loss functions.

This module also contains subclasses, such as :class:`Linear`, that function
as building blocks for more complex networks.
"""

from collections import OrderedDict
import pickle
import sys

import numpy as np
import theano
from theano.ifelse import ifelse
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano import tensor as T

from . import init
from . import search
from .fun import train_mode, function
from .utils import expand_to_batch, softmax_masked



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

    def summarize(self, grads, f=sys.stdout):
        def tensor_stats(m):
            return ', '.join([
                'norm = %g' % np.sqrt((m**2).sum()),
                'maxabs = %g' % np.abs(m).max(),
                'minabs = %g' % np.abs(m).min()])
        def summarize_parameter(name, p, g):
            print('%s\n    parameter %s\n    gradient %s' % (
                name, tensor_stats(p), tensor_stats(g)),
                file=f)
        params = list(self.parameters())
        assert len(grads) == len(params)
        for (name, p), grad in zip(params, grads):
            summarize_parameter('.'.join(name), p.get_value(borrow=True), grad)
        f.flush()

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

    This layer creates one shared parameter, `w` of shape
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
    dropout : float
        Dropout factor (the default value of 0 means dropout is not used).
    layernorm : bool
        If ``True``, layer normalization is used on the activations.
    """
    def __init__(self, name, input_dims, output_dims,
                 w=None, w_init=None, w_regularizer=None,
                 b=None, b_init=None, b_regularizer=None,
                 use_bias=True, dropout=0, layernorm=False):
        super().__init__(name)

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.use_bias = use_bias
        self.dropout = dropout
        self.layernorm = layernorm

        if w_init is None: w_init = init.Gaussian(fan_in=input_dims)
        if b_init is None: b_init = init.Constant(0.0)

        self.param('w', (input_dims, output_dims), init_f=w_init, value=w)
        self.regularize(self._w, w_regularizer)
        if use_bias:
            self.param('b', (output_dims,), init_f=b_init, value=b)
            self.regularize(self._b, b_regularizer)
        if dropout:
            self.add(Dropout('dropout', dropout))
        if layernorm:
            self.add(LayerNormalization('ln', (None, output_dims)))

    def __call__(self, inputs):
        outputs = T.dot(inputs, self._w)
        if self.layernorm: outputs = self.ln(outputs)
        if self.use_bias: outputs = outputs + self._b
        if self.dropout: outputs = self.dropout(outputs)
        return outputs


class Embeddings(Model):
    """Embeddings layer.

    This layer creates one shared parameter, `w` of shape
    `(alphabet_size, embedding_dims)`.

    Parameters
    ----------
    name : str
        Name of layer.
    embedding_dims : int
        Dimensionality of embeddings.
    alphabet_size : int
        Size of symbol alphabet.
    w : :class:`theano.compile.sharedvalue.SharedVariable`
        Weight vector to use, or pass ``None`` (default) to create a new
        one.
    w_init : :class:`.init.InitializationFunction`
        Initialization for weight vector, in case `w` is ``None``.
    w_regularizer : :class:`.regularize.Regularizer`, optional
        Regularization for weight matrix.
    dropout : float
        Dropout factor (the default value of 0 means dropout is not used).
    """
    def __init__(self, name, embedding_dims, alphabet_size,
                 w=None, w_init=None, w_regularizer=None,
                 dropout=0):
        super().__init__(name)

        self.embedding_dims = embedding_dims
        self.alphabet_size = alphabet_size
        self.dropout = dropout

        if w_init is None: w_init = init.Gaussian(fan_in=embedding_dims)

        self.param('w',
                (alphabet_size, embedding_dims), init_f=w_init, value=w)
        self.regularize(self._w, w_regularizer)
        if dropout:
            self.add(Dropout('dropout', dropout, sequence=True))

    def __call__(self, inputs):
        outputs = self._w[inputs]
        if self.dropout: outputs = self.dropout(outputs)
        return outputs


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
    """Long Short-Term Memory.

    layernorm : str
        One of `'ba1'` (eq 20--22 of Ba et al.), `'ba2'` (eq 29--31) or
        `False` (no layer normalization).
    """

    def __init__(self, name, input_dims, state_dims,
                 w=None, w_init=None, w_regularizer=None,
                 u=None, u_init=None, u_regularizer=None,
                 b=None, b_init=None, b_regularizer=None,
                 attention_dims=None, attended_dims=None,
                 layernorm='ba1'):
        super().__init__(name)

        assert layernorm in (False, 'ba1', 'ba2')
        assert (attention_dims is None) == (attended_dims is None)

        if attended_dims is not None:
            input_dims += attended_dims

        self.input_dims = input_dims
        self.state_dims = state_dims
        self.layernorm = layernorm
        self.attention_dims = attention_dims
        self.attended_dims = attended_dims
        self.use_attention = attention_dims is not None

        if w_init is None: w_init = init.Concatenated(
            [init.Gaussian(fan_in=input_dims)] * 4, axis=1)

        if u_init is None: u_init = init.Concatenated(
            [init.Orthogonal()]*4, axis=1)

        if b_init is None: b_init = init.Concatenated(
            [init.Constant(x) for x in [0.0, 1.0, 0.0, 0.0]])

        self.param('w', (input_dims, state_dims*4), init_f=w_init, value=w)
        self.param('u', (state_dims, state_dims*4), init_f=u_init, value=u)
        self.param('b', (state_dims*4,), init_f=b_init, value=b)

        if self.use_attention:
            self.add(Linear('attention_u', attended_dims, attention_dims))
            self.param('attention_w', (state_dims, attention_dims),
                       init_f=init.Gaussian(fan_in=state_dims))
            self.param('attention_v', (attention_dims,),
                       init_f=init.Gaussian(fan_in=attention_dims))
            self.regularize(self._attention_w, w_regularizer)
            if layernorm:
                self.add(LayerNormalization('ln_a', (None, attention_dims)))

        self.regularize(self._w, w_regularizer)
        self.regularize(self._u, u_regularizer)
        self.regularize(self._b, b_regularizer)

        if layernorm == 'ba1':
            self.add(LayerNormalization('ln_1', (None, state_dims*4)))
            self.add(LayerNormalization('ln_2', (None, state_dims*4)))
        if layernorm:
            self.add(LayerNormalization('ln_h', (None, state_dims)))

    def __call__(self, inputs, h_tm1, c_tm1,
                 attended=None, attended_dot_u=None, attention_mask=None):
        if self.use_attention:
            # Non-precomputed part of the attention vector for this time step
            #   _ x batch_size x attention_dims
            h_dot_w = T.dot(h_tm1, self._attention_w)
            if self.layernorm == 'ba1': h_dot_w = self.ln_a(h_dot_w)
            h_dot_w = h_dot_w.dimshuffle('x',0,1)
            # Attention vector, with distributions over the positions in
            # attended. Elements that fall outside the sentence in each batch
            # are set to zero.
            #   sequence_length x batch_size
            attention = softmax_masked(
                    T.dot(
                        T.tanh(attended_dot_u + h_dot_w),
                        self._attention_v).T,
                    attention_mask.T).T
            # Compressed attended vector, weighted by the attention vector
            #   batch_size x attended_dims
            compressed = (attended * attention.dimshuffle(0,1,'x')).sum(axis=0)
            # Append the compressed vector to the inputs and continue as usual
            inputs = T.concatenate([inputs, compressed], axis=1)
        if self.layernorm == 'ba1':
            x = (self.ln_1(T.dot(inputs, self._w)) +
                 self.ln_2(T.dot(h_tm1, self._u)))
        else:
            x = T.dot(inputs, self._w) + T.dot(h_tm1, self._u)
        x = x + self._b.dimshuffle('x', 0)
        def x_part(i): return x[:, i*self.state_dims:(i+1)*self.state_dims]
        i = T.nnet.sigmoid(x_part(0))
        f = T.nnet.sigmoid(x_part(1))
        o = T.nnet.sigmoid(x_part(2))
        c = T.tanh(        x_part(3))
        c_t = f*c_tm1 + i*c
        h_t = o*T.tanh(self.ln_h(c_t) if self.layernorm else c_t)
        if self.use_attention:
            return h_t, c_t, attention
        else:
            return h_t, c_t


class LSTMSequence(Model):
    def __init__(self, name, backwards, *args,
                 dropout=0, trainable_initial=False, offset=0, **kwargs):
        super().__init__(name)
        self.backwards = backwards
        self.trainable_initial = trainable_initial
        self.offset = offset
        self._step_fun = None
        self._attention_u_fun = None

        self.add(Dropout('dropout', dropout))
        self.add(LSTM('gate', *args, **kwargs))
        if self.trainable_initial:
            self.param('h_0', (self.gate.state_dims,),
                       init_f=init.Gaussian(fan_in=self.gate.state_dims))
            self.param('c_0', (self.gate.state_dims,),
                       init_f=init.Gaussian(fan_in=self.gate.state_dims))

    def step(self, inputs, inputs_mask, h_tm1, c_tm1, h_mask, *non_sequences):
        if self.gate.use_attention:
            # attended is the
            #   src_sequence_length x batch_size x attention_dims
            # matrix which we have attention on.
            #
            # attended_dot_u is the h_t-independent part of the final
            # attention vectors, which is precomputed for efficiency.
            #
            # attention_mask is a binary mask over the valid elements of
            # attended, which in practice is the same as the mask passed to
            # the encoder that created attended. Size
            #   src_sequence_length x batch_size
            h_t, c_t, attention = self.gate(
                    inputs, h_tm1 * h_mask.astype(theano.config.floatX), c_tm1,
                    attended=non_sequences[0],
                    attended_dot_u=non_sequences[1],
                    attention_mask=non_sequences[2])
            return (T.switch(inputs_mask.dimshuffle(0, 'x'), h_t, h_tm1),
                    T.switch(inputs_mask.dimshuffle(0, 'x'), c_t, c_tm1),
                    attention)
        else:
            h_t, c_t = self.gate(
                    inputs, h_tm1 * h_mask.astype(theano.config.floatX), c_tm1)
            return (T.switch(inputs_mask.dimshuffle(0, 'x'), h_t, h_tm1),
                    T.switch(inputs_mask.dimshuffle(0, 'x'), c_t, c_tm1))

    def step_fun(self):
        if self._step_fun is None:
            inputs = T.matrix('inputs')
            h_tm1 = T.matrix('h_tm1')
            c_tm1 = T.matrix('c_tm1')
            if self.gate.use_attention:
                attended=T.tensor3('attended')
                attended_dot_u=T.tensor3('attended_dot_u')
                attention_mask=T.matrix('attention_mask')
                self._step_fun = function(
                        [inputs, h_tm1, c_tm1,
                            attended, attended_dot_u, attention_mask],
                        self.step(inputs, T.ones(inputs.shape[:-1]),
                                  h_tm1, c_tm1, T.ones_like(h_tm1),
                                  attended, attended_dot_u, attention_mask))
            else:
                self._step_fun = function(
                        [inputs, h_tm1, c_tm1],
                        self.step(inputs, T.ones(inputs.shape[:-1]),
                                  h_tm1, c_tm1, T.ones_like(h_tm1)))
        return self._step_fun

    def attention_u_fun(self):
        assert self.gate.use_attention
        if self._attention_u_fun is None:
            attended = T.tensor3('attended')
            self._attention_u_fun = function(
                    [attended], self.gate.attention_u(attended))
        return self._attention_u_fun

    def search(self, predict_fun, embeddings,
               start_symbol, stop_symbol, max_length,
               h_0=None, c_0=None, attended=None, attention_mask=None,
               beam_size=4):
        if self.gate.use_attention:
            attended_dot_u = self.attention_u_fun()(attended)
        if self.trainable_initial:
            if h_0 is None:
                h_0 = self._h_0.get_value()[None,:]
            if c_0 is None:
                c_0 = self._c_0.get_value()[None,:]

        def step(i, states, outputs, outputs_mask):
            if self.gate.use_attention:
                result = self.step_fun()(
                        embeddings[outputs[-1]], states[0], states[1],
                        attended, attended_dot_u, attention_mask)
            else:
                result = self.step_fun()(
                        embeddings[outputs[-1]], states[0], states[1])
            h_t, c_t = result[:2]
            return [h_t, c_t], predict_fun(h_t)

        return search.beam(
                step, [h_0, c_0], h_0.shape[0], start_symbol, stop_symbol,
                max_length, beam_size=beam_size)


    def __call__(self, inputs, inputs_mask, h_0=None, c_0=None,
                 attended=None, attention_mask=None):
        if self.trainable_initial:
            batch_size = inputs.shape[1]
            if h_0 is None:
                h_0 = expand_to_batch(self._h_0, batch_size)
            if c_0 is None:
                c_0 = expand_to_batch(self._c_0, batch_size)
        attention_info = []
        if self.gate.use_attention:
            attention_info = [attended, self.gate.attention_u(attended),
                              attention_mask]
        dropout_masks = [self.dropout.mask(h_0.shape)]
        seqs, _ = theano.scan(
                fn=self.step,
                go_backwards=self.backwards,
                sequences=[{'input': inputs, 'taps': [self.offset]},
                           {'input': inputs_mask, 'taps': [self.offset]}],
                outputs_info=[h_0, c_0] + \
                             [None]*(1 if self.gate.use_attention else 0),
                non_sequences=dropout_masks + attention_info + \
                              self.gate.parameters_list())
        if self.backwards:
            return tuple(seq[::-1] for seq in seqs)
        else:
            return seqs


class Dropout(Model):
    """Dropout layer.
    
    name : str
        Name of layer.
    dropout : float
        Dropout factor (equivalent to 1 - retention probability)
    sequence : bool
        If True, dropout is not performed on the last dimension. This is
        useful for e.g. embedded symbol sequences, where either a symbol is
        kept intact or it is completely zeroed out.
    """

    def __init__(self, name, dropout, sequence=False):
        super().__init__(name)
        self.p = 1.0 - dropout
        self.rng = RandomStreams()
        self.sequence = sequence

    def mask(self, shape):
        """Return a scaled mask for a (symbolic) shape.

        This can be used for dropout in recurrent layers, where a fixed mask
        is passed through the non_sequences argument to theano.scan().
        """
        if self.p == 1: return T.ones(shape)
        if self.sequence:
            m = T.shape_padright(self.rng.binomial(shape[:-1], p=self.p)
                    ).astype(theano.config.floatX)
        else:
            m = self.rng.binomial(shape, p=self.p).astype(theano.config.floatX)
        return m / self.p

    def __call__(self, inputs):
        if self.p == 1: return inputs
        m = self.mask(inputs.shape)
        return ifelse(train_mode, inputs * m, inputs)

        
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

        mean = inputs.mean(axis=self.axis, keepdims=True).astype(
                theano.config.floatX)
        std = inputs.std(axis=self.axis, keepdims=True).astype(
                theano.config.floatX)
        normed = (inputs - mean) / (std + self.epsilon)
        return normed * self._g.dimshuffle(*broadcast)

