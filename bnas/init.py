"""Initialization functions.

Each initialization function is implemented as a class, with callable objects
that map shape tuples to initialized numpy arrays.
"""

import numpy as np
import theano


class InitializationFunction:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class Constant(InitializationFunction):
    def __init__(self, a):
        """Constant initialization.

        All elements are initialized with the same constant value.

        Parameters
        ----------
        a : float
            Initialization constant.
        """
        self.a = a

    def __call__(self, dims, dtype=theano.config.floatX):
        return np.full(dims, a, dtype=dtype)


class Gaussian(InitializationFunction):
    def __init__(self, mean=0.0, dev=0.01):
        """Isotropic Gaussian initialization.

        Elements are initialized independently from a Gaussian distribution
        with a given mean and standard deviation.

        Parameters
        ----------
        mean : float
            Mean of distribution (default: 0)
        dev : float
            Standard deviation of distribution (default: 0.01)
        """
        self.mean = mean
        self.dev = dev

    def __call__(self, dims, dtype=theano.config.floatX):
        m = np.random.standard_normal(dims)*self.dev + self.mean
        return me.astype(dtype)


class Uniform(InitializationFunction):
    def __init__(self, mean=0.0, scale=0.01):
        """Uniform initialization.

        Elements are initialized independently from a uniform distribution
        with a given mean and scale (so the minimum value is `mean - scale/2`
        and the maximum is `mean + scale/2`).

        Parameters
        ----------
        mean : float
            Mean of distribution (default: 0)
        scale : float
            Scale of distribution (default: 0.01)
        """
        self.mean = mean
        self.scale = scale

    def __call__(self, dims, dtype=theano.config.floatX):
        return np.random.uniform(
                size=dims,
                low=mean-0.5*scale,
                high=mean+0.5*scale).astype(config.floatX)


class Orthogonal(InitializationFunction):
    def __init__(self, scale=1.0):
        """Orthogonal initalization.

        Create an orthogonal matrix, i.e. where M x M.T == I.
        Note that this can only be used with 2D matrixes, and typically with
        square ones.

        Parameters
        ----------
        scale : float
            Scaling factor (default: 1.0)
        """
        self.scale = scale

    def __call__(self, dims, dtype=theano.config.floatX):
        # Code from Blocks: blocks/blocks/initialization.py
        if len(dims) != 2:
            raise ValueError(
                    'Orthogonal matrix must be square, '
                    'but has shape: %s' % dims)
        if dims[0] == dims[1]:
            M = gaussian(dims)
            Q, R = np.linalg.qr(M)
            Q = Q * np.sign(np.diag(R))
            return Q * self.scale
        M1 = gaussian((dims[0], dims[0]))
        M2 = gaussian((dims[1], dims[1]))
        Q1, R1 = np.linalg.qr(M1)
        Q2, R2 = np.linalg.qr(M2)
        Q1 = Q1 * np.sign(np.diag(R1))
        Q2 = Q2 * np.sign(np.diag(R2))
        n_min = min(dims)
        return np.dot(Q1[:, :n_min], Q2[:n_min, :]) * self.scale


class Identity(InitializationFunction):
    def __init__(self, scale=1.0):
        """Identity matrix initialization.

        Creates a scaled identity matrix, which must be 2D and square.

        Parameters
        ----------
        scale : float
            Scale (default: 1)
        """
        self.scale = scale

    def __call__(self, dims, dtype=theano.config.floatX):
        if len(dims) != 2:
            raise ValueError(
                    'Identity matrix must be 2D, but has shape %s' % dims)
        if dims[0] != dims[1]:
            raise ValueError(
                    'Identity matrix must be square, but has shape %s' % dims)
        return np.eye(dims[0], dtype=dtype)

