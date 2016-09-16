"""Multilingual character-based language model using LSTMs"""

from pprint import pprint
import pickle
import sys
import os

import numpy as np
import theano
from theano import tensor as T

from bnas.model import Model, Linear, Embeddings, Sequence, LinearSelection
from bnas.optimize import Adam, iterate_batches
from bnas.init import Gaussian, Orthogonal, Concatenated, Constant
from bnas.utils import softmax_3d, expand_to_batch
from bnas.loss import batch_sequence_crossentropy
from bnas.text import TextEncoder
from bnas.fun import function

# TODO: proper evaluation where multilingual model is evaluated on same
# monolingual test sets as the monolingual models.
class SelectionLSTM(Model):
    def __init__(self, name, input_dims, state_dims, parallel_dims, pick_dims,
                 layernorm=False):
        super().__init__(name)
        self.n_states = 2
        self.use_attention = False

        self.input_dims = input_dims
        self.state_dims = state_dims
        self.parallel_dims = parallel_dims
        self.pick_dims = pick_dims

        w_init = Concatenated(
                [Gaussian(fan_in=input_dims)] * 3, axis=1)
        u_init = Concatenated(
                [Orthogonal()] * 3, axis=1)
        b_init = Concatenated(
                [Constant(x) for x in [0.0, 1.0, 0.0]])

        self.add(LinearSelection(
            'selection',
            state_dims+input_dims, state_dims, pick_dims, parallel_dims,
            w_init=Concatenated([
                Concatenated(
                    [Orthogonal(), Gaussian(fan_in=input_dims)],
                    div_fun=lambda n: [state_dims, input_dims])
                for _ in range(parallel_dims)],
                axis=1),
            sw_init=Concatenated([
                Concatenated(
                    [Orthogonal(), Gaussian(fan_in=input_dims+pick_dims)],
                    div_fun=lambda n: [state_dims, input_dims+pick_dims])
                for _ in range(parallel_dims)],
                axis=1),
            input_select=True))

        self.param('w', (input_dims, state_dims*3), init_f=w_init)
        self.param('u', (state_dims, state_dims*3), init_f=u_init)
        self.param('b', (state_dims*3,), init_f=b_init)

    def __call__(self, inputs_pick, h_tm1, c_tm1):
        inputs = inputs_pick[:, :self.input_dims]
        pick = inputs_pick[:, self.input_dims:]
        x = T.dot(inputs, self._w) + T.dot(h_tm1, self._u)
        x = x + self._b.dimshuffle('x', 0)
        def x_part(i): return x[:, i*self.state_dims:(i+1)*self.state_dims]
        i = T.nnet.sigmoid(x_part(0))
        f = T.nnet.sigmoid(x_part(1))
        o = T.nnet.sigmoid(x_part(2))
        c = self.selection(T.concatenate([h_tm1, inputs], axis=-1), pick)
        c_t = f*c_tm1 + i*c
        h_t = o*T.tanh(c_t)
        return h_t, c_t


class LanguageModel(Model):
    def __init__(self, name, config):
        super().__init__(name)

        self.config = config

        pprint(config)
        sys.stdout.flush()

        self.add(Embeddings(
            'lang_embeddings',
            len(config['langs']), config['lang_embedding_dims']))
        self.add(Embeddings(
            'ortho_embeddings',
            len(config['langs']), config['lang_embedding_dims']))
        self.add(Embeddings(
            'embeddings', len(config['encoder']), config['embedding_dims']))
        self.add(LinearSelection(
            'hidden',
            config['state_dims'],
            config['embedding_dims'],
            config['lang_embedding_dims'],
            config['parallel_dims'],
            dropout=config['dropout'],
            layernorm=config['layernorm'],
            input_select=True))
        #self.add(Linear(
        #    'hidden',
        #    config['state_dims'] + config['lang_embedding_dims'],
        #    config['embedding_dims'],
        #    dropout=config['dropout'],
        #    layernorm=config['layernorm']))
        self.add(Linear(
            'emission',
            config['embedding_dims'], len(config['encoder']),
            w=self.embeddings._w.T))
        self.add(Sequence(
            'decoder', SelectionLSTM, False,
            config['embedding_dims'], config['state_dims'],
            config['parallel_dims'], config['lang_embedding_dims'],
            dropout=config['recurrent_dropout'],
            layernorm=config['recurrent_layernorm'],
            trainable_initial=True, offset=-1))

        #h_t = T.matrix('h_t')
        #self.predict_fun = function(
        #        [h_t],
        #        T.nnet.softmax(self.emission(T.tanh(self.hidden(h_t)))))

    #def step(self, inputs, inputs_mask, h_tm1, c_tm1, lang, *non_sequences):
    #    h_t, c_t = self.gate(inputs, h_tm1, c_tm1, lang)
    #    return (T.switch(inputs_mask.dimshuffle(0, 'x'), h_t, h_tm1),
    #            T.switch(inputs_mask.dimshuffle(0, 'x'), c_t, c_tm1))

    def __call__(self, lang, outputs, outputs_mask):
        lang_embedded = T.shape_padleft(self.lang_embeddings(lang)).repeat(
                outputs.shape[0], axis=0)
        outputs_embedded = self.embeddings(outputs)
        h_seq, c_seq = self.decoder(
                T.concatenate([outputs_embedded, lang_embedded], axis=-1),
                outputs_mask)
        pred_seq = softmax_3d(self.emission(
            T.tanh(self.hidden(
                h_seq, self.ortho_embeddings(lang), sequence=True))))
        return h_seq, pred_seq

    def cross_entropy(self, lang, outputs, outputs_mask):
        # Construct a Theano expression for computing the cross-entropy of an
        # example with respect to the current model predictions.
        _, symbol_seq = self(lang, outputs, outputs_mask)
        return batch_sequence_crossentropy(
                symbol_seq, outputs[1:], outputs_mask[1:])
 
    def loss(self, lang, outputs, outputs_mask):
        # Construct a Theano expression for computing the loss function used
        # during training. This consists of cross-entropy loss for the
        # training batch plus regularization terms.
        return super().loss() + self.cross_entropy(lang, outputs, outputs_mask)

    """
    TODO: need to extend LSTMSequence.search() with optional fixed vector to
    append to each input.

    def search(self, lang, max_length):
        # Get the parameter values of the embeddings and initial state.
        embeddings = self.embeddings._w.get_value(borrow=True)

        # Perform a beam search.
        return self.decoder.search(
                self.predict_fun, embeddings,
                self.config['index']['<S>'], self.config['index']['</S>'],
                max_length)
    """

def read_file(filename):
    print('Reading %s...' % filename, flush=True)
    def parse_line(line):
        fields = line.lower().split('\t')
        if len(fields) == 2: return fields[1].strip()
        return None

    with open(filename, 'r', encoding='utf-8') as f:
        lines = [parse_line(line) for line in f if not line.startswith('#')]
        lines = [line for line in lines if line]

    lang = os.path.basename(filename)[:3]
    name = os.path.splitext(os.path.basename(filename))[0][4:]

    return (lang, name), lines


if __name__ == '__main__':
    import os
    from time import time

    model_filename = sys.argv[1]

    if os.path.exists(model_filename):
        with open(model_filename, 'rb') as f:
            config = pickle.load(f)
            lm = LanguageModel('lm', config)
            lm.load(f)
            #encoder = config['encoder']
            print('Load model from %s' % model_filename)
            m = lm.lang_embeddings._w.get_value()
            from scipy.cluster.hierarchy import single, average, dendrogram
            from scipy.spatial.distance import pdist
            from matplotlib import pyplot as plt
            y = pdist(m, 'cosine')
            z = average(y) # single(y)
            dendrogram(z, labels=config['langs'])
            plt.show()
    else:
        with open(sys.argv[2], 'r') as f:
            filenames = f.read().split()
        print('%d texts to read...' % len(filenames))

        files_lines = list(map(read_file, filenames))
        langs = sorted({lang for (lang,_),_ in files_lines})
        lang_idx = {lang:i for i,lang in enumerate(langs)}

        lang_lines = [(lang_idx[lang], line) for (lang,_),lines in files_lines
                                             for line in lines]

        encoder = TextEncoder(
                sequences=[line for _,line in lang_lines],
                special=('<S>', '</S>'))

        print('Read %d sentences, %d symbols' % (
                len(lang_lines), len(encoder)),
              flush=True)

        # Model hyperparameters
        config = {
                'langs': langs,
                'encoder': encoder,
                'max_length': 256,
                'lang_embedding_dims': 32,
                'parallel_dims': 8,
                'embedding_dims': 32,
                'state_dims': 512,
                'recurrent_layernorm': 'ba1',
                'recurrent_dropout': 0.3,
                'layernorm': True,
                'dropout': 0.3
                }

        lm = LanguageModel('lm', config)

        # Training-specific parameters
        n_epochs = 20
        batch_size = 128
        test_size = batch_size

        # Create the model.
        sym_lang = T.lvector('lang')
        sym_outputs = T.lmatrix('outputs')
        sym_outputs_mask = T.bmatrix('outputs_mask')

        # Create an optimizer instance, manually specifying which
        # parameters to optimize, which loss function to use, which inputs
        # (none) and outputs are used for the model. We also specify the
        # gradient clipping threshold.
        optimizer = Adam(
                lm.parameters(),
                lm.loss(sym_lang, sym_outputs, sym_outputs_mask),
                [sym_lang], [sym_outputs, sym_outputs_mask],
                grad_max_norm=5.0)

        # Compile a function to compute cross-entropy of a batch.
        cross_entropy = function(
                [sym_lang, sym_outputs, sym_outputs_mask],
                lm.cross_entropy(sym_lang, sym_outputs, sym_outputs_mask))

        test_set = lang_lines[:test_size]
        train_set = lang_lines[test_size:]

        # Get one batch of testing data, encoded as a masked matrix.
        test_outputs, test_outputs_mask = encoder.pad_sequences(
                [line for _,line in test_set],
                max_length=config['max_length'])
        test_lang = np.array([lang for lang,_ in test_set], dtype=np.int32)

        batch_nr = 0
        sent_nr = 0
        for i in range(n_epochs):
            for batch in iterate_batches(
                    train_set, batch_size, lambda t: len(t[1])):
                outputs, outputs_mask = encoder.pad_sequences(
                        [line for _,line in batch],
                        max_length=config['max_length'])
                lang = np.array([lang for lang,_ in batch], dtype=np.int32)
                if batch_nr % 10 == 0:
                    test_loss = cross_entropy(
                            test_lang, test_outputs, test_outputs_mask)
                    test_loss_bit = (
                            (test_size/test_outputs_mask[1:].sum())*
                            test_loss/(np.log(2)))
                    print('Test loss: %.3f bits/char' % test_loss_bit)
                t0 = time()
                loss = optimizer.step(lang, outputs, outputs_mask)
                t = time() - t0

                if np.isnan(loss):
                    print('NaN at iteration %d!' % (i+1))
                    break
                print(('Batch %d:%d:%d: train: %.3f bits/char (%.2f s)') % (
                    i+1, batch_nr+1, sent_nr+1,
                    (batch_size/outputs_mask[1:].sum())*loss/np.log(2),
                    t),
                    flush=True)

                batch_nr += 1
                sent_nr += len(batch)
        
            with open('%s.%d' % (model_filename, i+1), 'wb') as f:
                pickle.dump(config, f)
                lm.save(f)
                print('Saved model to %s.%d' % (model_filename, i+1))


        with open(model_filename, 'wb') as f:
            pickle.dump(config, f)
            lm.save(f)
            print('Saved model to %s' % model_filename)

    #pred, pred_mask, scores = lm.search(128)
    #
    #for sent, score in zip(pred, scores):
    #    print(score, ''.join(symbols[x] for x in sent.flatten()))

