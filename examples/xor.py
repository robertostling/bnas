"""Example using an overly complex MLP to learn the XOR function.

The point of this is to demonstrate and test some of the more advanced
functions, like normalization and dropout layers, which don't really make
sense with a network this simple.
"""

import numpy as np
import theano
from theano import tensor as T

from bnas.model import Model, Linear
from bnas.optimize import Adam, SGD
from bnas.init import Gaussian
from bnas.regularize import L2
from bnas.fun import function

class MLP(Model):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

        self.add(Linear('hidden', 2, 8,
                        use_bias=True,
                        w_regularizer=L2(0.001),
                        layernorm=True,
                        dropout=0.1))
        self.add(Linear('output', 8, 1))

    def loss(self, inputs, outputs):
        loss = super().loss()
        return loss + ((self(inputs) - outputs) ** 2).mean()

    def __call__(self, inputs):
        return T.nnet.sigmoid(self.output(T.tanh(self.hidden(inputs))))


if __name__ == '__main__':
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=theano.config.floatX)
    y = np.array([[0],    [1],    [1],    [0]],    dtype=theano.config.floatX)

    x = x*0.8 + 0.1
    y = y*0.8 + 0.1

    inputs = T.matrix('inputs')
    outputs = T.matrix('outputs')
    xor = MLP('xor')
    optimizer = Adam(xor.parameters(), xor.loss(inputs, outputs),
                     [inputs], [outputs])

    for i in range(1000):
        loss = optimizer.step(x, y)
        if np.isnan(loss):
            print('NaN at iteration %d!' % (i+1))
            break

    print('Last loss = %g. Predictions vs targets:' % loss)

    predict = function([inputs], xor(inputs), name='XOR_predict')
    print(np.hstack([predict(x), y]))

