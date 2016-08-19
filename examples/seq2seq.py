from time import time
import numpy as np
import theano
from theano import tensor as T

from bnas.model import Model, Linear, LSTM, LayerNormalization
from bnas.optimize import Adam
from bnas.init import Gaussian, Orthogonal, Constant
from bnas.regularize import L2
from bnas.utils import expand_to_batch
from bnas.loss import batch_sequence_crossentropy
from bnas.text import encode_sequences, mask_sequences
from bnas.search import beam, greedy
from bnas.fun import function


class EncoderGate(Model):
    def __init__(self, name, embedding_dims, state_dims, n_symbols, **kwargs):
        super().__init__(name, **kwargs)

        # Define the parameters required for a recurrent transition using
        # LSTM units, taking a character embedding as input and outputting 
        # (through a fully connected tanh layer) a distribution over symbols.
        # The embeddings are shared between the input and output.
        self.param('embeddings', (n_symbols, embedding_dims),
                   init_f=Gaussian(fan_in=embedding_dims))
        self.add(LSTM('transition', embedding_dims, state_dims*2,
                 use_layernorm=True))

    def __call__(self, symbol, mask, state, *non_sequences):
        # Construct the Theano symbol expressions for the new state and the
        # output predictions, given the embedded current symbol and the
        # previous state.
        return T.switch(mask.dimshuffle(0,'x'),
                        self.transition(symbol, state),
                        state)


class Encoder(Model):
    def __init__(self, name, embedding_dims, state_dims, n_symbols, **kwargs):
        super().__init__(name, **kwargs)

        self.state_dims = state_dims
        self.n_symbols = n_symbols

        # Import the parameters of the recurrence into the main model.
        self.add(EncoderGate('gate', embedding_dims, state_dims, n_symbols))

        self.param('state0', (state_dims*2,),
                   init_f=Gaussian(fan_in=state_dims*2))

        inputs = T.lmatrix('inputs')
        inputs_mask = T.bmatrix('inputs_mask')
        self.encode = function([inputs, inputs_mask],
                               self(inputs, inputs_mask)[-1])

    def __call__(self, inputs, inputs_mask):
        # Construct the Theano symbolic expression for the predicted state
        # sequence, which basically amounts to calling
        # theano.scan() using the EncoderGate instance as inner function.
        batch_size = inputs.shape[1]
        embedded_inputs = self.gate._embeddings[inputs]
        state_seq, _ = theano.scan(
                fn=self.gate,
                sequences=[embedded_inputs, inputs_mask],
                outputs_info=[expand_to_batch(self._state0, batch_size)],
                non_sequences=self.gate.parameters_list())
        return state_seq


class DecoderGate(Model):
    def __init__(self, name, embedding_dims, state_dims, n_symbols, **kwargs):
        super().__init__(name, **kwargs)

        self.state_dims = state_dims
        # Define the parameters required for a recurrent transition using
        # LSTM units, taking a character embedding as input and outputting 
        # (through a fully connected tanh layer) a distribution over symbols.
        # The embeddings are shared between the input and output.
        self.param('embeddings', (n_symbols, embedding_dims),
                   init_f=Gaussian(fan_in=embedding_dims))
        self.add(LSTM('transition', embedding_dims, state_dims*2,
                 use_layernorm=True))
        self.add(Linear('hidden', state_dims, embedding_dims))
        self.add(Linear('emission', embedding_dims, n_symbols,
                        w=self._embeddings.T))

    def __call__(self, last, state, *non_sequences):
        # Construct the Theano symbol expressions for the new state and the
        # output predictions, given the embedded previous symbol and the
        # previous state.
        new_state = self.transition(last, state)
        h = new_state[:, :self.state_dims]
        return (new_state,
                T.nnet.softmax(self.emission(T.tanh(self.hidden(h)))))


class Decoder(Model):
    def __init__(self, name, embedding_dims, state_dims, n_symbols, **kwargs):
        super().__init__(name, **kwargs)

        self.state_dims = state_dims
        self.n_symbols = n_symbols

        # Import the parameters of the recurrence into the main model.
        self.add(DecoderGate('gate', embedding_dims, state_dims, n_symbols))

        # Compile a function for a single recurrence step, this is used during
        # decoding (but not during training).
        last = T.matrix('last')
        state = T.matrix('state')
        self.step = function([last, state], self.gate(last, state))

    def __call__(self, input, outputs, outputs_mask):
        # Construct the Theano symbolic expression for the state and output
        # prediction sequences, which basically amounts to calling
        # theano.scan() using the DecoderGate instance as inner function.
        batch_size = outputs.shape[1]
        embedded_outputs = self.gate._embeddings[outputs] \
                         * outputs_mask.dimshuffle(0,1,'x')
        (state_seq, symbol_seq), _ = theano.scan(
                fn=self.gate,
                sequences=[{'input': embedded_outputs, 'taps': [-1]}],
                outputs_info=[input, None],
                non_sequences=self.gate.parameters_list())
        return state_seq, symbol_seq

    def cross_entropy(self, input, outputs, outputs_mask):
        # Construct a Theano expression for computing the cross-entropy of an
        # example with respect to the current model predictions.
        state_seq, symbol_seq = self(input, outputs, outputs_mask)
        batch_size = outputs.shape[1]
        return batch_sequence_crossentropy(
                symbol_seq, outputs[1:], outputs_mask[1:])
 
    def loss(self, input, outputs, outputs_mask):
        # Construct a Theano expression for computing the loss function used
        # during training. This consists of cross-entropy loss for the
        # training batch plus regularization terms.
        #
        # Get an expression for parameter-wise regularization terms.
        loss = super().loss()
        state_seq, symbol_seq = self(input, outputs, outputs_mask)
        batch_size = outputs.shape[1]
        xent = batch_sequence_crossentropy(
                symbol_seq, outputs[1:], outputs_mask[1:])
        return loss + xent

    def search(self, input, start_symbol, stop_symbol,
               max_length, min_length):
        # Perform a beam search.

        batch_size = input.shape[0]

        # Get the parameter values of the embeddings.
        embeddings = self.gate._embeddings.get_value(borrow=True)

        # Define a step function, which takes a list of states and a history
        # of previous outputs, and returns the next states and output
        # predictions.
        def step(i, states, outputs, outputs_mask):
            # In this case we only condition the step on the last output,
            # and there is only one state.
            state, outputs = self.step(embeddings[outputs[-1]], states[0])
            return [state], outputs

        # Call the library beam search function to do the dirty job.
        return beam(step, [input], batch_size, start_symbol,
                    stop_symbol, max_length, min_length=min_length)


class Seq2Seq(Model):
    def __init__(self, name, embedding_dims, state_dims,
                 n_enc_symbols, n_dec_symbols, **kwargs):
        super().__init__(name, **kwargs)

        self.add(Encoder('encoder', embedding_dims, state_dims, n_enc_symbols))
        self.add(Decoder('decoder', embedding_dims, state_dims, n_dec_symbols))

    def __call__(self, inputs, inputs_mask, outputs, outputs_mask):
        v = self.encoder(inputs[::-1,:], inputs_mask[::-1,:])
        pred_outputs = self.decoder(v[-1], outputs, outputs_mask)

    def loss(self, inputs, inputs_mask, outputs, outputs_mask):
        v = self.encoder(inputs[::-1,:], inputs_mask[::-1,:])
        return self.decoder.loss(v[-1], outputs, outputs_mask)

    def search(self, inputs, inputs_mask, start_symbol,
               stop_symbol, max_length, min_length):
        return self.decoder.search(
                self.encoder.encode(inputs[::-1,:], inputs_mask[::-1,:]),
                start_symbol, stop_symbol, max_length, min_length)

    def cross_entropy(self, inputs, inputs_mask, outputs, outputs_mask):
        v = self.encoder(inputs[::-1,:], inputs_mask[::-1,:])
        return self.decoder.cross_entropy(v[-1], outputs, outputs_mask)


if __name__ == '__main__':
    import sys
    import os

    corpus_filename = sys.argv[1]
    model_filename = sys.argv[2]
    assert os.path.exists(corpus_filename)

    def parse_pair(line):
        pair = line.split(' ||| ')
        assert len(pair) == 2
        return (pair[0].strip(), pair[1].strip())

    with open(corpus_filename, 'r', encoding='utf-8') as f:
        pairs = [parse_pair(line) for line in f]

    src_symbols, src_index, src_encoded = encode_sequences(
            [pair[0] for pair in pairs])
    trg_symbols, trg_index, trg_encoded = encode_sequences(
            [pair[1] for pair in pairs])

    print('Read %d pairs' % len(pairs), flush=True)
    print('Source alphabet: %d' % len(src_symbols), flush=True)
    print('Target alphabet: %d' % len(trg_symbols), flush=True)

    # Create the model.
    inputs = T.lmatrix('inputs')
    inputs_mask = T.bmatrix('inputs_mask')
    outputs = T.lmatrix('outputs')
    outputs_mask = T.bmatrix('outputs_mask')

    seq2seq = Seq2Seq('seq2seq', 32, 512, len(src_symbols), len(trg_symbols))

    if os.path.exists(model_filename):
        with open(model_filename, 'rb') as f:
            seq2seq.load(f)
            print('Load model from %s' % model_filename, flush=True)
    else:
        # Create an Adam optimizer instance, manually specifying which
        # parameters to optimize, which loss function to use, which inputs
        # (none) and outputs are used for the model. We also specify the
        # gradient clipping threshold.
        optimizer = Adam(
                seq2seq.parameters(),
                seq2seq.loss(inputs, inputs_mask, outputs, outputs_mask),
                [inputs, inputs_mask], [outputs, outputs_mask],
                grad_max_norm=5.0)

        # Compile a function to compute cross-entropy of a batch.
        cross_entropy = function(
                [inputs, inputs_mask, outputs, outputs_mask],
                seq2seq.cross_entropy(
                    inputs, inputs_mask, outputs, outputs_mask))

        batch_size = 128
        test_size = 32
        max_length = 128

        # Get one batch of testing data, encoded as a masked matrix.
        test_inputs, test_inputs_mask = mask_sequences(
                src_encoded[:test_size], max_length)
        test_outputs, test_outputs_mask = mask_sequences(
                trg_encoded[:test_size], max_length)

        n_batches = 0
        for i in range(1):
            for j in range(test_size, len(src_encoded), batch_size):
                # Create one training batch
                src_batch = src_encoded[j:j+batch_size]
                trg_batch = trg_encoded[j:j+batch_size]
                inputs, inputs_mask = mask_sequences(src_batch, max_length)
                outputs, outputs_mask = mask_sequences(trg_batch, max_length)
                if n_batches % 10 == 0:
                    test_loss = cross_entropy(
                            test_inputs, test_inputs_mask,
                            test_outputs, test_outputs_mask)
                    print('Batch %d:%d: test: %.3f b/char' % (
                        i+1, j+1,
                        (test_size/test_outputs_mask[1:].sum())
                            *test_loss/(np.log(2))),
                        flush=True)

                t0 = time()
                loss = optimizer.step(inputs, inputs_mask,
                                      outputs, outputs_mask)
                t = time() - t0

                if np.isnan(loss):
                    print('NaN at iteration %d!' % (i+1))
                    break
                print('Batch %d:%d: train: %.3f b/char (%.3f s)' % (
                    i+1, j+1,
                    (batch_size/outputs_mask[1:].sum())*loss/np.log(2),
                    t),
                    flush=True)

                n_batches += 1

                if n_batches % 100 == 1:
                    pred, pred_mask, scores = seq2seq.search(
                            test_inputs, test_inputs_mask,
                            trg_index['<S>'], trg_index['</S>'], 128, 32)

                    for src_sent, sent, score in zip(
                            test_inputs.T, pred[-1].T, scores[-1]):
                        print(''.join(
                            src_symbols[x] for x in src_sent.flatten()
                            if x != 1))
                        print(score, ''.join(
                            trg_symbols[x] for x in sent.flatten()))
                        print('-'*72, flush=True)

        with open(model_filename, 'wb') as f:
            seq2seq.save(f)
            print('Saved model to %s' % model_filename)


