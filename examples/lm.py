"""Character-based language model using LSTM.

This example demonstrates recurrent networks with BNAS, using an LSTM with
variational dropout (Gal 2016) and layer normalization (Ba, Kiros and Hinton
2016).

Usage:
    python3 lm.py model.bin corpus.txt

If model.bin exists, the corpus filename can be left out to generate some
test sentences.
"""

import pickle

import numpy as np
import theano
from theano import tensor as T

from bnas.model import Model, Linear, Embeddings, LSTMSequence
from bnas.optimize import Nesterov, iterate_batches
from bnas.init import Gaussian
from bnas.utils import softmax_3d
from bnas.loss import batch_sequence_crossentropy
from bnas.text import TextEncoder
from bnas.fun import function


class LanguageModel(Model):
    def __init__(self, name, config):
        super().__init__(name)

        self.config = config

        self.add(Embeddings(
            'embeddings', config['n_symbols'], config['embedding_dims']))
        self.add(Linear(
            'hidden',
            config['state_dims'], config['embedding_dims'],
            dropout=config['dropout'],
            layernorm=config['layernorm']))
        self.add(Linear(
            'emission',
            config['embedding_dims'], config['n_symbols'],
            w=self.embeddings._w.T))
        self.add(LSTMSequence(
            'decoder', False,
            config['embedding_dims'], config['state_dims'],
            dropout=config['recurrent_dropout'],
            layernorm=config['recurrent_layernorm'],
            trainable_initial=True, offset=-1))

        h_t = T.matrix('h_t')
        self.predict_fun = function(
                [h_t],
                T.nnet.softmax(self.emission(T.tanh(self.hidden(h_t)))))

    def __call__(self, outputs, outputs_mask):
        h_seq, c_seq = self.decoder(
                self.embeddings(outputs), outputs_mask)
        pred_seq = softmax_3d(self.emission(T.tanh(self.hidden(h_seq))))
        return h_seq, pred_seq

    def cross_entropy(self, outputs, outputs_mask):
        # Construct a Theano expression for computing the cross-entropy of an
        # example with respect to the current model predictions.
        _, symbol_seq = self(outputs, outputs_mask)
        return batch_sequence_crossentropy(
                symbol_seq, outputs[1:], outputs_mask[1:])
 
    def loss(self, outputs, outputs_mask):
        # Construct a Theano expression for computing the loss function used
        # during training. This consists of cross-entropy loss for the
        # training batch plus regularization terms.
        return super().loss() + self.cross_entropy(outputs, outputs_mask)

    def search(self, max_length):
        # Get the parameter values of the embeddings and initial state.
        embeddings = self.embeddings._w.get_value(borrow=True)

        # Perform a beam search.
        return self.decoder.search(
                self.predict_fun, embeddings,
                self.config['index']['<S>'], self.config['index']['</S>'],
                max_length)


if __name__ == '__main__':
    import sys
    import os
    from time import time

    model_filename = sys.argv[1]

    if os.path.exists(model_filename):
        with open(model_filename, 'rb') as f:
            config = pickle.load(f)
            lm = LanguageModel('lm', config)
            lm.load(f)
            encoder = config['encoder']
            print('Load model from %s' % model_filename)
    else:
        corpus_filename = sys.argv[2]
        assert os.path.exists(corpus_filename)

        # Model hyperparameters
        config = {
                'max_length': 192,
                'embedding_dims': 64,
                'state_dims': 1024,
                'recurrent_layernorm': 'ba1',
                'recurrent_dropout': 0.2,
                'layernorm': True,
                'dropout': 0.5
                }

        with open(corpus_filename, 'r', encoding='utf-8') as f:
            sents = [line.strip() for line in f
                     if len(line) >= 10 and len(line) <= config['max_length']]

        encoder = TextEncoder(sequences=sents, special=('<S>', '</S>'))

        config['n_symbols'] = len(encoder)
        config['encoder'] = encoder

        lm = LanguageModel('lm', config)

        # Training-specific parameters
        n_epochs = 100
        batch_size = 64
        test_size = batch_size
        batch_nr = 0

        # Create the model.
        sym_outputs = T.lmatrix('outputs')
        sym_outputs_mask = T.bmatrix('outputs_mask')

        # Create an optimizer instance, manually specifying which
        # parameters to optimize, which loss function to use, which inputs
        # (none) and outputs are used for the model. We also specify the
        # gradient clipping threshold.
        optimizer = Nesterov(
                lm.parameters(),
                lm.loss(sym_outputs, sym_outputs_mask),
                [], [sym_outputs, sym_outputs_mask],
                learning_rate=0.02,
                grad_max_norm=5.0)

        # Compile a function to compute cross-entropy of a batch.
        cross_entropy = function(
                [sym_outputs, sym_outputs_mask],
                lm.cross_entropy(sym_outputs, sym_outputs_mask))

        test_set = sents[:test_size]
        train_set = sents[test_size:]

        # Get one batch of testing data, encoded as a masked matrix.
        test_outputs, test_outputs_mask = encoder.pad_sequences(test_set)

        for i in range(n_epochs):
            for batch in iterate_batches(train_set, batch_size, len):
                outputs, outputs_mask = encoder.pad_sequences(batch)
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
                print(('Batch %d:%d: train: %.3f bits/char (%.2f s)') % (
                    i+1, batch_nr+1,
                    (batch_size/outputs_mask[1:].sum())*loss/np.log(2),
                    t),
                    flush=True)

                batch_nr += 1

        with open(model_filename, 'wb') as f:
            pickle.dump(config, f)
            lm.save(f)
            print('Saved model to %s' % model_filename)

    pred, pred_mask, scores = lm.search(128)

    for sent, score in zip(pred, scores):
        print(score, ''.join(symbols[x] for x in sent.flatten()))

