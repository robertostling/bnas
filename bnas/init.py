"""Initialization functions.

Each initialization function is implemented as a class, with callable objects
that map shape tuples to initialized numpy arrays.

Examples
--------
>>> from bnas.init import Orthogonal
>>> from bnas.model import Linear
>>> Linear('transition', 100, 100, w_init=Orthogonal())
"""

import sys

import numpy as np
import theano


class InitializationFunction:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


def even_split(k):
    def split(n):
        if n % k != 0:
            raise ValueError('Can not split evenly!')
        return [n//k for _ in range(k)]
    return split


class Concatenated(InitializationFunction):
    """Concatenated initializations.

    This can be used to construct a matrix where different rows are
    initialized in different ways, for instance if you have a recurrent
    transformations with some inputs and want the state-to-state
    transformation to use orthogonal matrix initialization and the
    input-to-state transformation to use Gaussian initialization.

    Arguments
    ---------
    init_funs : list of functions
        Functions to initialize each of the matrices
    div_fun : function (int => list of int), optional
        Function mapping the shape of the first dimension to a list of
        first-dimension shapes for each of the initialization functions in
        `init_funs`. By default, the matrix is split up evenly, as in the
        example below.
    axis : int
        Axis along which to concatenate, default is 0 which is useful when
        transforming several inputs, but set to 1 if several concatenated
        outputs are produced.
    Example
    -------
    >>> Linear('transition', dims*2, dims, w_init=Concatenated([
    ...     Orthogonal(),
    ...     Gaussian(fan_in=dims)]))
    """
    def __init__(self, init_funs, div_fun=None, axis=0):
        if div_fun is None:
            k = len(init_funs)
            div_fun = even_split(k)
        self.init_funs = init_funs
        self.div_fun = div_fun
        self.axis = axis

    def __call__(self, dims, dtype=theano.config.floatX):
        divs = self.div_fun(dims[self.axis])
        assert sum(divs) == dims[self.axis]
        shapes = [dims[:self.axis] + (dim0,) + dims[self.axis+1:]
                  for dim0 in divs]
        return np.concatenate(
                [f(shape, dtype) for f, shape in zip(self.init_funs, shapes)],
                axis=self.axis)


class Constant(InitializationFunction):
    """Constant initialization.

    All elements are initialized with the same constant value.

    Parameters
    ----------
    a : float
        Initialization constant.
    """

    def __init__(self, a):
        self.a = a

    def __call__(self, dims, dtype=theano.config.floatX):
        return np.full(dims, self.a, dtype=dtype)


class Gaussian(InitializationFunction):
    """Isotropic Gaussian initialization.

    Elements are initialized independently from a Gaussian distribution
    with a given mean and standard deviation.

    Parameters
    ----------
    dev : float, optional
        Standard deviation of distribution
    mean : float, optional
        Mean of distribution
    fan_in : int, optional
        If this argument is given, `dev` and `mean` are ignored. Instead,
        mean 0 and standard deviation :math:`\sqrt{2/fan_in}` is used.
    """

    def __init__(self, dev=1.0, mean=0.0, fan_in=None):
        if fan_in is None:
            self.dev = dev
            self.mean = mean
        else:
            self.dev = np.sqrt(2.0/fan_in)
            self.mean = 0.0

    def __call__(self, dims, dtype=theano.config.floatX):
        m = np.random.standard_normal(dims)*self.dev + self.mean
        return m.astype(dtype)


class Uniform(InitializationFunction):
    """Uniform initialization.

    Elements are initialized independently from a uniform distribution
    with a given mean and scale (so the minimum value is `mean - scale/2`
    and the maximum is `mean + scale/2`).

    Parameters
    ----------
    scale : float, optional
        Scale of distribution
    mean : float, optional
        Mean of distribution
    """

    def __init__(self, scale=0.01, mean=0.0):
        self.scale = scale
        self.mean = mean

    def __call__(self, dims, dtype=theano.config.floatX):
        return np.random.uniform(
                size=dims,
                low=mean-0.5*scale,
                high=mean+0.5*scale).astype(config.floatX)


class Orthogonal(InitializationFunction):
    """Orthogonal initalization.

    Create an orthogonal matrix, i.e. where M x M.T == I.
    Note that this can only be used with 2D matrixes, and typically with
    square ones.

    Parameters
    ----------
    scale : float, optional
        Scaling factor
    """

    def __init__(self, scale=1.0):
        self.scale = scale

    def __call__(self, dims, dtype=theano.config.floatX):
        # Code from Blocks: blocks/blocks/initialization.py
        if len(dims) != 2:
            raise ValueError(
                    'Orthogonal matrix must be square, '
                    'but has shape: %s' % dims)
        if dims[0] == dims[1]:
            M = np.random.standard_normal(dims)
            Q, R = np.linalg.qr(M)
            Q = Q * np.sign(np.diag(R))
            return (Q * self.scale).astype(dtype)
        print('WARNING: creating non-square orthogonal matrix (bug?)',
              file=sys.stderr)
        M1 = np.random.standard_normal((dims[0], dims[0]))
        M2 = np.random.standard_normal((dims[1], dims[1]))
        Q1, R1 = np.linalg.qr(M1)
        Q2, R2 = np.linalg.qr(M2)
        Q1 = Q1 * np.sign(np.diag(R1))
        Q2 = Q2 * np.sign(np.diag(R2))
        n_min = min(dims)
        return (np.dot(Q1[:,:n_min], Q2[:n_min,:]) * self.scale).astype(dtype)


class Identity(InitializationFunction):
    """Identity matrix initialization.

    Creates a scaled identity matrix, which must be 2D and square.

    Parameters
    ----------
    scale : float, optional
        Scale
    """

    def __init__(self, scale=1.0):
        self.scale = scale

    def __call__(self, dims, dtype=theano.config.floatX):
        if len(dims) != 2:
            raise ValueError(
                    'Identity matrix must be 2D, but has shape %s' % dims)
        if dims[0] != dims[1]:
            raise ValueError(
                    'Identity matrix must be square, but has shape %s' % dims)
        return np.eye(dims[0], dtype=dtype)


class Identity2D(InitializationFunction):
    """Identity filter for 2D convolutional layers."""

    def __init__(self, scale=1.0):
        self.scale = scale

    def __call__(self, dims, dtype=theano.config.floatX):
        if not len(dims) == 4:
            raise ValueError('Shape of 2D filter must be 4D')
        if not (dims[0] == dims[1] and dims[-2] == dims[-1]):
            raise ValueError(
                'First/second and third/fourth dimensions must be equal')
        if not dims[-1] % 2 == 1:
            raise ValueError('Filter size must be odd')
        m = np.zeros(dims, dtype=dtype)
        for i in range(dims[0]):
            m[i,i,dims[-2]//2,dims[-1]//2] = self.scale
        return m

