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
    params : OrderedDict of str ->
             :class:`theano.compile.sharedvalue.SharedVariable`
        Mapping from parameter names to Theano shared variables. These are the
        trainable parameters of the model.
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

    def param(self, name, dims, init_f=init.Constant(np.nan),
              dtype=theano.config.floatX):
        """Create a new parameter (using a Theano shared variable)

        Parameters
        ----------
        name : str
            Name of parameter, this will be used directly in `self.params`
            and used to create `self._name`.
        dims : tuple
            Shape of the parameter vector.
        init_f : (tuple => numpy.ndarray)
            Function used to initialize the parameter vector.
        dtype : numpy.dtype
            Data type (default is `theano.config.floatX`)

        Returns
        -------
        p : :class:`theano.compile.sharedvalue.SharedVariable`
        """
        p = theano.shared(init_f(dims, dtype=dtype), name=name)
        self.params[name] = p
        setattr(self, '_'+name, p)
        return p

    def save(self, f):
        """Save the weights of this model to a file object.

        Parameters
        ----------
        f : file
            File object to write to, assumed to be opened in 'wb' mode.
        """
        pickle.dump({name: p.get_value() for name, p in self.params.items()},
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
        names = frozenset(data.keys()) & frozenset(self.params.keys())
        if not allow_incomplete and len(names) < len(self.params):
            raise ValueError(
                    'The following parameters are missing: %s' % ', '.join(
                        sorted(frozenset(self.params.keys()) - names)))
        if not allow_unused and len(names) < len(data):
            raise ValueError(
                    'The following parameters are unused: %s' % ', '.join(
                        sorted(frozenset(data.keys()) - names)))
        for name in names:
            value = data[name]
            old_value = self.params[name].get_value()
            if value.shape != old_value.shape:
                raise ValueError(
                        'Loaded shape is %s but %s expected' % (
                            value.shape, old_value.shape))
            shared.set_value(value)

    def linear(self, name, inputs, inputs_dims, outputs_dims,
               w=None, w_init=init.Gaussian(0.01),
               b=None, b_init=init.Constant(0),
               use_bias=True):
        """Create a fully connected linear layer.
        
        This layer creates one shared parameter, `name_w` of shape
        `(input_dims, output_dims)` if `use_bias` is ``False``, otherwise it
        also creates `name_b` of shape `output_dims` for biases.

        Parameters
        ----------
        name : str
            Name of layer.
        inputs : :class:`~tensor.TensorVariable`
            Inputs to layer, second last dimension of shape must be equal to
            `inputs_dims`.
        inputs_dims : int
            Number of inputs.
        outputs_dims : int
            Number of outputs.
        w : :class:`theano.compile.sharedvalue.SharedVariable`
            Weight vector to use, or pass ``None`` (default) to create a new
            one.
        w_init : :class:`.init.InitializationFunction`
            Initialization for weight vector, in case `w` is ``None``.
        b : :class:`theano.compile.sharedvalue.SharedVariable`
            Bias vector to use, or pass ``None`` (default) to create a new
            one.
        b_init : :class:`.init.InitializationFunction`
            Initialization for bias vector, in case `b` is ``None``.
        use_bias : bool
            If ``False``, no bias is used and the `b` and `b_init` parameters
            are ignored.

        Returns
        -------
        outputs : :class:`~tensor.TensorVariable`
            Symbolic variable for the layer's outputs. This will be of shape
            `inputs.shape[:-1] + (outputs_dims,)`.
        """

        if w is None:
            w = self.param(name+'_w', (inputs_dims, outputs_dims),
                           init_f=w_init)
        if use_bias:
            if b is None:
                b = self.param(name+'_b', (outputs_dims,), init_f=b_init)
            return T.dot(inputs, w) + b
        else:
            return T.dot(inputs, w)

