import numpy as np
import theano
from theano import tensor as T

from bnas.model import Model, Linear, DSGU
from bnas.optimize import Adam
from bnas.init import Gaussian, Orthogonal, Constant
from bnas.regularize import L2, StateNorm
from bnas.utils import expand_to_batch
from bnas.loss import batch_sequence_crossentropy
from bnas.text import encode_sequences, mask_sequences
from bnas.search import beam


class Gate(Model):
    def __init__(self, name, embedding_dims, state_dims, n_symbols, **kwargs):
        super().__init__(name, **kwargs)

        self.param('embeddings', (n_symbols, embedding_dims),
                   init_f=Gaussian(fan_in=embedding_dims))
        self.add(DSGU('transition', embedding_dims, state_dims))
        self.add(Linear('hidden', state_dims, embedding_dims))
        self.add(Linear('emission', embedding_dims, n_symbols,
                        w=self._embeddings.T))

    def __call__(self, last, state, *non_sequences):
        return (self.transition(last, state),
                T.nnet.softmax(self.emission(T.tanh(self.hidden(state)))))



class LanguageModel(Model):
    def __init__(self, name, embedding_dims, state_dims, n_symbols, **kwargs):
        super().__init__(name, **kwargs)

        self.state_dims = state_dims
        self.n_symbols = n_symbols

        self.add(Gate('gate', embedding_dims, state_dims, n_symbols))
        self.param('state0', (state_dims,), init_f=Gaussian(fan_in=state_dims))

        last = T.matrix('last')
        state = T.matrix('state')
        self.step = theano.function([last, state], self.gate(last, state))

    def __call__(self, outputs, outputs_mask):
        batch_size = outputs.shape[1]
        embedded_outputs = self.gate._embeddings[outputs] \
                         * outputs_mask.dimshuffle(0,1,'x')
        (state_seq, symbol_seq), _ = theano.scan(
                fn=self.gate,
                sequences=[{'input': embedded_outputs, 'taps': [-1]}],
                outputs_info=[expand_to_batch(self._state0, batch_size), None],
                non_sequences=self.gate.parameters_list())
        return state_seq, symbol_seq

    def cross_entropy(self, outputs, outputs_mask):
        state_seq, symbol_seq = self(outputs, outputs_mask)
        batch_size = outputs.shape[1]
        return batch_sequence_crossentropy(
                symbol_seq, outputs[1:], outputs_mask[1:])
 
    def loss(self, outputs, outputs_mask):
        loss = super().loss()
        state_seq, symbol_seq = self(outputs, outputs_mask)
        batch_size = outputs.shape[1]
        state_norm = StateNorm(50.0)(
                T.concatenate([
                    expand_to_batch(self._state0, batch_size
                        ).dimshuffle('x',0,1),
                    state_seq],
                    axis=0), outputs_mask)
        xent = batch_sequence_crossentropy(
                symbol_seq, outputs[1:], outputs_mask[1:])
        return loss + xent + state_norm

    def search(self, batch_size, start_symbol, stop_symbol, max_length):
        embeddings = self.gate._embeddings.get_value(borrow=True)
        state0 = np.repeat(self._state0.get_value()[None,:],
                           batch_size, axis=0)

        def step(i, states, outputs, outputs_mask):
            state, outputs = self.step(embeddings[outputs[-1]], states[0])
            return [state], outputs

        return beam(step, [state0], batch_size, start_symbol,
                    stop_symbol, max_length)

    def state_norm(self, outputs, outputs_mask):
        loss = super().loss()
        state_seq, symbol_seq = self(outputs, outputs_mask)
        state_norm = StateNorm(50.0)(state_seq, outputs_mask[1:])
        return state_norm


if __name__ == '__main__':
    import sys
    import os

    corpus_filename = sys.argv[1]
    model_filename = sys.argv[2]
    assert os.path.exists(corpus_filename)
    assert not os.path.exists(model_filename)

    with open(corpus_filename, 'r', encoding='utf-8') as f:
        sents = [line.strip() for line in f if len(line) >= 10]

    symbols, index, encoded = encode_sequences(sents)
    n_symbols = len(symbols)

    outputs = T.lmatrix('outputs')
    outputs_mask = T.matrix('outputs_mask')
    lm = LanguageModel('lm', 32, 256, n_symbols)
    optimizer = Adam(lm.parameters(), lm.loss(outputs, outputs_mask),
                     [], [outputs, outputs_mask])

    gradients = theano.function(
            [outputs, outputs_mask], list(optimizer.grad.values()))
    state_norm = theano.function(
            [outputs, outputs_mask], lm.state_norm(outputs, outputs_mask))
    cross_entropy = theano.function(
            [outputs, outputs_mask], lm.cross_entropy(outputs, outputs_mask))

    batch_size = 64
    test_outputs, test_outputs_mask = mask_sequences(encoded[:batch_size], 256)
    for i in range(1):
        #for j in range(batch_size, len(encoded), batch_size):
        for j in range(batch_size, batch_size*5, batch_size):
            batch = encoded[j:j+batch_size]
            outputs, outputs_mask = mask_sequences(batch, 256)
            #print('StateNorm: %g' % state_norm(outputs, outputs_mask))
            #for name, p in lm.parameters():
            #    print('Norm of',  '.'.join(name),
            #            np.sqrt((p.get_value(borrow=True)**2).sum()))
            #for g, (name, _) in zip(gradients(outputs, outputs_mask), lm.parameters()):
            #    print('Gradient norm of', '.'.join(name), np.sqrt((g**2).sum()))
            test_loss = cross_entropy(test_outputs, test_outputs_mask)
            loss = optimizer.step(outputs, outputs_mask)
            if np.isnan(loss):
                print('NaN at iteration %d!' % (i+1))
                break
            print('Epoch %d sentence %d: loss = %g, test xent = %g' % (
                i+1, j+1, loss/np.log(2),
                test_loss/(np.log(2))))

    pred, pred_mask, scores = lm.search(1, index['<S>'], index['</S>'], 40)
    #print(pred)
    #print(pred_mask)
    #print(pred.flatten())
    #print(len(symbols))
    print(''.join(symbols))
    print(scores)
    for sent in pred:
        print(''.join(symbols[x] for x in sent.flatten()))

    with open(model_filename, 'wb') as f:
        lm.save(f)
        print('Saved model to %s' % model_filename)

