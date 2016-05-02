import numpy as np
import theano
from theano import tensor as T

from bnas.model import Model, Linear
from bnas.optimize import Adam, SGD
from bnas.init import Gaussian
from bnas.regularize import L2

class MLP(Model):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

        self.inputs = T.matrix('inputs')
        self.outputs = T.matrix('outputs')

        hidden = self.add(
            Linear('hidden', 2, 8,
                   w_regularizer=L2(0.001)))
        output = self.add(Linear('output', 8, 1))

        self.pred_outputs = T.nnet.sigmoid(
                output(T.tanh(hidden(self.inputs))))

        self.predict = theano.function([self.inputs], self.pred_outputs)

    def loss(self):
        loss = super().loss()
        return loss + ((self.pred_outputs - self.outputs) ** 2).mean()


if __name__ == '__main__':
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=theano.config.floatX)
    y = np.array([[0],    [1],    [1],    [0]],    dtype=theano.config.floatX)

    xor = MLP('xor')
    optimizer = Adam(xor.parameters(), xor.loss(), [xor.inputs], [xor.outputs])

    for i in range(1000):
        loss = optimizer.step(x, y)
        if np.isnan(loss):
            print('NaN at iteration %d!' % (i+1))
            break

    print('Last loss = %g. Predictions vs targets:' % loss)
    print(np.hstack([xor.predict(x), y]))

