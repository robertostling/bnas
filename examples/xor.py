import numpy as np
import theano
from theano import tensor as T

from bnas.model import Model
from bnas.train import Adam
from bnas.init import Gaussian

class XOR(Model):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

        self.inputs = T.matrix('inputs')
        self.outputs = T.matrix('outputs')

        hidden = T.tanh(
                self.linear('hidden', self.inputs, 2, 8))
        self.pred_outputs = T.nnet.sigmoid(
                self.linear('pred_outputs', hidden, 8, 1))

        self.predict = theano.function([self.inputs], self.pred_outputs)

    def loss(self):
        return ((self.pred_outputs - self.outputs) ** 2).mean()


if __name__ == '__main__':
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=theano.config.floatX)
    y = np.array([[0],    [1],    [1],    [0]],    dtype=theano.config.floatX)

    xor = XOR('xor')
    optimizer = Adam(xor.params, xor.loss(), [xor.inputs], [xor.outputs])

    for i in range(1000):
        loss = optimizer.step(x, y)

    print('Last loss = %g. Predictions vs targets:' % loss)
    print(np.hstack([xor.predict(x), y]))

