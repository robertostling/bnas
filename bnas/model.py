from collections import OrderedDict
import pickle

import theano

import init


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

    def __init__(self, name, default_init=init.Gaussian(dev=0.01)):
        """Initialize an empty model.

        Parameters
        ----------
        name : str
            Name of the model.
        default_init : (tuple => :class:`numpy.ndarray`)
            Function used to initialize the parameter vector. The default is
            an isotropic Gaussian distribution with zero mean and 0.01
            standard deviation.
        """
        self.name = name
        self.params = OrderedDict()
        self.default_init = default_init

    def param(self, name, dims, init_f=None, dtype=theano.config.floatX):
        """Create a new parameter (using a Theano shared variable)

        Parameters
        ----------
        name : str
            Name of parameter, this will be used directly in `self.params`
            and used to create `self._name`.
        dims : tuple
            Shape of the parameter vector.
        init_f : (tuple => numpy.ndarray)
            Function used to initialize the parameter vector, if none is
            specified `self.default_init` will be used.
        dtype : numpy.dtype
            Data type (default is `theano.config.floatX`)

        Returns
        -------
        p : :class:`theano.compile.sharedvalue.SharedVariable`
        """
        if init_f is None: init_f = self.default_init
        p = theano.shared(init_f(dims, dtype=dtype), name=name)
        self.params[name] = p
        setattr(self, '_'+name, p)
        return p

    def save(self, f):
        """Save the weights of this model to a file object

        Parameters
        ----------
        f : file
            File object to write to, assumed to be opened in 'wb' mode.
        """
        pickle.dump({name: p.get_value() for name, p in self.params.items()},
                    f, -1)

