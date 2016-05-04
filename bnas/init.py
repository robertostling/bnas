"""Initialization functions.

Each initialization function is implemented as a class, with callable objects
that map shape tuples to initialized numpy arrays.

Examples
--------
>>> from bnas.init import Orthogonal
>>> from bnas.model import Linear
>>> Linear('transition', 100, 100, w_init=Orthogonal())
"""

import numpy as np
import theano


class InitializationFunction:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


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
    dev : float
        Standard deviation of distribution
    mean : float
        Mean of distribution
    """

    def __init__(self, dev=0.01, mean=0.0):
        self.dev = dev
        self.mean = mean

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
    scale : float
        Scale of distribution
    mean : float
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
    scale : float
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
    scale : float
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

