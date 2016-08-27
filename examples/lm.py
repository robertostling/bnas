import random

import numpy as np
import theano
from theano import tensor as T

from bnas.model import (Model, Linear, LSTM, RHN, Dropout, LayerNormalization,
                        RandomStreams)
from bnas.optimize import Adam
from bnas.init import Gaussian, Orthogonal, Constant
from bnas.regularize import L2
from bnas.utils import expand_to_batch
from bnas.loss import batch_sequence_crossentropy
from bnas.text import encode_sequences, mask_sequences
from bnas.search import beam, greedy
from bnas.fun import function

GateType = LSTM

class Gate(Model):
    def __init__(self, name, embedding_dims, state_dims, n_symbols, **kwargs):
        super().__init__(name, **kwargs)

        states_dims = state_dims*2 if GateType is LSTM else state_dims
        self.state_dims = state_dims
        self.states_dims = states_dims
        # Define the parameters required for a recurrent transition using
        # GRU/LSTM units, taking a character embedding as input and outputting 
        # (through a fully connected tanh layer) a distribution over symbols.
        # The embeddings are shared between the input and output.
        self.param('embeddings', (n_symbols, embedding_dims),
                   init_f=Gaussian(fan_in=embedding_dims))
        self.add(GateType('transition', embedding_dims, states_dims,
                 use_layernorm=True, dropout=0.3))
        self.add(Linear('hidden', state_dims, embedding_dims))
        self.add(Linear('emission', embedding_dims, n_symbols,
                        w=self._embeddings.T))

    def __call__(self, last, state, *non_sequences):
        # Construct the Theano symbol expressions for the new state and the
        # output predictions, given the embedded previous symbol and the
        # previous state.
        new_state = self.transition(last, state, *non_sequences)
        h = new_state
        if self.state_dims != self.states_dims:
            h = new_state[:, :self.state_dims]
        return (new_state,
                T.nnet.softmax(self.emission(T.tanh(self.hidden(h)))))


class LanguageModel(Model):
    def __init__(self, name, embedding_dims, state_dims, n_symbols, **kwargs):
        super().__init__(name, **kwargs)

        states_dims = state_dims*2 if GateType is LSTM else state_dims
        self.state_dims = state_dims
        self.states_dims = states_dims
        self.n_symbols = n_symbols

        # Import the parameters of the recurrence into the main model.
        self.add(Gate('gate', embedding_dims, state_dims, n_symbols))
        # Add a learnable parameter for the initial state.
        self.param('state0', (states_dims,),
                   init_f=Gaussian(fan_in=states_dims))

        # Compile a function for a single recurrence step, this is used during
        # decoding (but not during training).
        last = T.matrix('last')
        state = T.matrix('state')
        self.step = function([last, state], self.gate(last, state))

    def __call__(self, outputs, outputs_mask):
        # Construct the Theano symbolic expression for the state and output
        # prediction sequences, which basically amounts to calling
        # theano.scan() using the Gate instance as inner function.
        batch_size = outputs.shape[1]
        embedded_outputs = self.gate._embeddings[outputs] \
                         * outputs_mask.dimshuffle(0,1,'x')
        state0 = expand_to_batch(self._state0, batch_size)
        (state_seq, symbol_seq), _ = theano.scan(
                fn=self.gate,
                sequences=[{'input': embedded_outputs, 'taps': [-1]}],
                outputs_info=[state0, None],
                non_sequences=
                    self.gate.transition.dropout_masks(
                        embedded_outputs.shape[1:],
                        state0[:, :self.state_dims].shape) +
                    self.gate.parameters_list())
        return state_seq, symbol_seq

    def cross_entropy(self, outputs, outputs_mask):
        # Construct a Theano expression for computing the cross-entropy of an
        # example with respect to the current model predictions.
        state_seq, symbol_seq = self(outputs, outputs_mask)
        batch_size = outputs.shape[1]
        return batch_sequence_crossentropy(
                symbol_seq, outputs[1:], outputs_mask[1:])
 
    def loss(self, outputs, outputs_mask):
        # Construct a Theano expression for computing the loss function used
        # during training. This consists of cross-entropy loss for the
        # training batch plus regularization terms.
        #
        # Get an expression for parameter-wise regularization terms.
        loss = super().loss()
        state_seq, symbol_seq = self(outputs, outputs_mask)
        batch_size = outputs.shape[1]
        xent = batch_sequence_crossentropy(
                symbol_seq, outputs[1:], outputs_mask[1:])
        return loss + xent

    def search(self, batch_size, start_symbol, stop_symbol,
               max_length, min_length):
        # Perform a beam search.

        # Get the parameter values of the embeddings and initial state.
        embeddings = self.gate._embeddings.get_value(borrow=True)
        state0 = np.repeat(self._state0.get_value()[None,:],
                           batch_size, axis=0)

        # Define a step function, which takes a list of states and a history
        # of previous outputs, and returns the next states and output
        # predictions.
        def step(i, states, outputs, outputs_mask):
            # In this case we only condition the step on the last output,
            # and there is only one state.
            state, outputs = self.step(embeddings[outputs[-1]], states[0])
            return [state], outputs

        # Call the library beam search function to do the dirty job.
        return beam(step, [state0], batch_size, start_symbol,
                    stop_symbol, max_length, min_length=min_length)


if __name__ == '__main__':
    import sys
    import os
    from time import time

    corpus_filename = sys.argv[1]
    model_filename = sys.argv[2]
    assert os.path.exists(corpus_filename)

    with open(corpus_filename, 'r', encoding='utf-8') as f:
        sents = [line.strip() for line in f if len(line) >= 10]

    # Create a vocabulary table+index and encode the input sentences.
    symbols, index, encoded = encode_sequences(sents)
    n_symbols = len(symbols)

    # Create the model.
    outputs = T.lmatrix('outputs')
    outputs_mask = T.bmatrix('outputs_mask')
    lm = LanguageModel('lm', 32, 512, n_symbols)

    if os.path.exists(model_filename):
        with open(model_filename, 'rb') as f:
            lm.load(f)
            print('Load model from %s' % model_filename)
    else:
        # Create an Adam optimizer instance, manually specifying which
        # parameters to optimize, which loss function to use, which inputs
        # (none) and outputs are used for the model. We also specify the
        # gradient clipping threshold.
        optimizer = Adam(lm.parameters(), lm.loss(outputs, outputs_mask),
                         [], [outputs, outputs_mask],
                         grad_max_norm=5.0)

        # Compile a function to compute cross-entropy of a batch.
        cross_entropy = function(
                [outputs, outputs_mask],
                lm.cross_entropy(outputs, outputs_mask))

        batch_size = 128
        test_size = 128
        max_length = 128
        batch_nr = 0

        # Get one batch of testing data, encoded as a masked matrix.
        test_outputs, test_outputs_mask = mask_sequences(
                encoded[:test_size], max_length)

        order = list(range(test_size, len(encoded)))

        for i in range(100):
            random.shuffle(order)
            for j in range(0, len(order), batch_size):
                # Create one training batch
                batch = [encoded[k] for k in order[j:j+batch_size]]
                outputs, outputs_mask = mask_sequences(batch, max_length)
                if batch_nr % 10 == 0:
                    test_loss = cross_entropy(test_outputs, test_outputs_mask)
                    test_loss_bit = (
                            (test_size/test_outputs_mask[1:].sum())*
                            test_loss/(np.log(2)))
                    print('Test loss: %.3f bits/char' % test_loss_bit)
                t0 = time()
                loss = optimizer.step(outputs, outputs_mask)
                t = time() - t0

                if np.isnan(loss):
                    print('NaN at iteration %d!' % (i+1))
                    break
                print(('Batch %d:%d: train: %.3f b/char (%.2f s)') % (
                    i+1, j+1,
                    (batch_size/outputs_mask[1:].sum())*loss/np.log(2),
                    t),
                    flush=True)

                batch_nr += 1

        with open(model_filename, 'wb') as f:
            lm.save(f)
            print('Saved model to %s' % model_filename)

    pred, pred_mask, scores = lm.search(
            1, index['<S>'], index['</S>'], 72, 72)

    for sent, score in zip(pred, scores):
        print(score, ''.join(symbols[x] for x in sent.flatten()))

