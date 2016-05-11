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

        # Define the parameters required for a recurrent transition using
        # DSGU units, taking a character embedding as input and outputting 
        # (through a fully connected tanh layer) a distribution over symbols.
        # The embeddings are shared between the input and output.
        self.param('embeddings', (n_symbols, embedding_dims),
                   init_f=Gaussian(fan_in=embedding_dims))
        self.add(DSGU('transition', embedding_dims, state_dims))
        self.add(Linear('hidden', state_dims, embedding_dims))
        self.add(Linear('emission', embedding_dims, n_symbols,
                        w=self._embeddings.T))

    def __call__(self, last, state, *non_sequences):
        # Construct the Theano symbol expressions for the new state and the
        # output predictions, given the embedded previous symbol and the
        # previous state.
        return (self.transition(last, state),
                T.nnet.softmax(self.emission(T.tanh(self.hidden(state)))))


class LanguageModel(Model):
    def __init__(self, name, embedding_dims, state_dims, n_symbols, **kwargs):
        super().__init__(name, **kwargs)

        self.state_dims = state_dims
        self.n_symbols = n_symbols

        # Import the parameters of the recurrence into the main model.
        self.add(Gate('gate', embedding_dims, state_dims, n_symbols))
        # Add a learnable parameter for the initial state.
        self.param('state0', (state_dims,), init_f=Gaussian(fan_in=state_dims))

        # Compile a function for a single recurrence step, this is used during
        # decoding (but not during training).
        last = T.matrix('last')
        state = T.matrix('state')
        self.step = theano.function([last, state], self.gate(last, state))

    def __call__(self, outputs, outputs_mask):
        # Construct the Theano symbolic expression for the state and output
        # prediction sequences, which basically amounts to calling
        # theano.scan() using the Gate instance as inner function.
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
        # Use hidden state norm regularization.
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
                    stop_symbol, max_length)


if __name__ == '__main__':
    import sys
    import os

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
    outputs_mask = T.matrix('outputs_mask')
    lm = LanguageModel('lm', 32, 256, n_symbols)

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
        cross_entropy = theano.function(
                [outputs, outputs_mask],
                lm.cross_entropy(outputs, outputs_mask))

        batch_size = 128
        # Get one batch of testing data, encoded as a masked matrix.
        test_outputs, test_outputs_mask = mask_sequences(
                encoded[:batch_size], 256)
        for i in range(1):
            for j in range(batch_size, len(encoded), batch_size):
                # Create one training batch
                batch = encoded[j:j+batch_size]
                outputs, outputs_mask = mask_sequences(batch, 256)
                test_loss = cross_entropy(test_outputs, test_outputs_mask)
                loss = optimizer.step(outputs, outputs_mask)
                if np.isnan(loss):
                    print('NaN at iteration %d!' % (i+1))
                    break
                print('Batch %d:%d: train loss = %g, test xent = %g' % (
                    i+1, j+1, loss/np.log(2),
                    test_loss/(np.log(2))))

        with open(model_filename, 'wb') as f:
            lm.save(f)
            print('Saved model to %s' % model_filename)

    pred, pred_mask, scores = lm.search(1, index['<S>'], index['</S>'], 72)

    for sent, score in zip(pred, scores):
        print(score, ''.join(symbols[x] for x in sent.flatten()))
