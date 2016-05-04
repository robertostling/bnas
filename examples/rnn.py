import numpy as np
import theano
from theano import tensor as T

from bnas.model import Model, Linear
from bnas.optimize import Adam, SGD
from bnas.init import Gaussian, Orthogonal, Constant
from bnas.regularize import L2
from bnas.utils import expand_to_batch
from bnas.loss import batch_sequence_crossentropy
from bnas.text import encode_sequences, mask_sequences


class Gate(Model):
    def __init__(self, name, state_dims, n_symbols, **kwargs):
        super().__init__(name, **kwargs)

        self.add(Linear('transition', state_dims, state_dims,
                        w_init=Orthogonal()))
        self.add(Linear('emission', state_dims, n_symbols,
                        w_init=Gaussian(0.01)))

    # TODO: add feedback, then implement greedy + beam search.
    #       Make sure feedback doesn't mess up anything later.
    def __call__(self, state, *non_sequences):
        return (T.tanh(self.transition(state)),
                T.nnet.softmax(self.emission(state)))


class RNN(Model):
    def __init__(self, name, state_dims, n_symbols, **kwargs):
        super().__init__(name, **kwargs)

        self.state_dims = state_dims
        self.n_symbols = n_symbols

        self.add(Gate('gate', state_dims, n_symbols))
        self.param('state0', (state_dims,), init_f=Constant(0.0))

    def __call__(self, n_steps, batch_size):
        (state_seq, symbol_seq), _ = theano.scan(
                fn=self.gate,
                n_steps=n_steps,
                outputs_info=[expand_to_batch(self._state0, batch_size), None],
                non_sequences=self.gate.parameters_list())
        return state_seq, symbol_seq

    def loss(self, outputs, outputs_mask):
        loss = super().loss()
        seq_length, batch_size = outputs.shape
        state_seq, symbol_seq = self(seq_length, batch_size)
        # TODO: try StateNorm(penalty)(state_seq)
        xent = batch_sequence_crossentropy(symbol_seq, outputs, outputs_mask)
        return loss + xent


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'r', encoding='utf-8') as f:
        sents = [line.strip() for line in f if len(line) >= 10]

    symbols, index, encoded = encode_sequences(sents)
    n_symbols = len(symbols)

    outputs = T.lmatrix('outputs')
    outputs_mask = T.matrix('outputs_mask')
    lm = RNN('lm', 256, n_symbols)
    optimizer = Adam(lm.parameters(), lm.loss(outputs, outputs_mask),
                     [], [outputs, outputs_mask])

    batch_size = 64
    test_outputs, test_outputs_mask = mask_sequences(encoded[:batch_size], 256)
    for i in range(1):
        for j in range(batch_size, len(encoded), batch_size):
            batch = encoded[j:j+batch_size]
            outputs, outputs_mask = mask_sequences(batch, 256)
            test_loss = optimizer.get_loss(test_outputs, test_outputs_mask)
            loss = optimizer.step(outputs, outputs_mask)
            if np.isnan(loss):
                print('NaN at iteration %d!' % (i+1))
                break
            print('Epoch %d sentence %d: loss = %g, test loss = %g' % (
                i+1, j+1, loss, test_loss))

