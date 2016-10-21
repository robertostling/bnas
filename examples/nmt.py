"""Neural machine translation example.

This is a pretty straightforward implementation of Badhanau et al. (2014):
https://arxiv.org/pdf/1409.0473v7.pdf

The system is word based and assumes tokenized inputs, the only bells and
whistles are variational dropout and layer normalization. Otherwise this is
meant as a starting point for NMT experiments.
"""

import numpy as np
import theano
from theano import tensor as T

from bnas.model import Model, Linear, LSTM, Sequence
from bnas.optimize import Adam, iterate_batches
from bnas.init import Gaussian
from bnas.utils import softmax_3d
from bnas.loss import batch_sequence_crossentropy
from bnas.text import TextEncoder
from bnas.fun import function


class NMT(Model):
    def __init__(self, name, config):
        super().__init__(name)
        self.config = config

        self.param('src_embeddings',
                   (len(config['src_encoder']), config['src_embedding_dims']),
                   init_f=Gaussian(fan_in=config['src_embedding_dims']))
        self.param('trg_embeddings',
                   (len(config['trg_encoder']), config['trg_embedding_dims']),
                   init_f=Gaussian(fan_in=config['trg_embedding_dims']))
        self.add(Linear('hidden',
                        config['decoder_state_dims'],
                        config['trg_embedding_dims']))
        self.add(Linear('emission',
                        config['trg_embedding_dims'],
                        len(config['trg_encoder']),
                        w=self._trg_embeddings.T))
        for prefix, backwards in (('fwd', False), ('back', True)):
            self.add(Sequence(
                prefix+'_encoder', LSTM, backwards,
                config['src_embedding_dims'] + (
                    config['encoder_state_dims'] if backwards else 0),
                config['encoder_state_dims'],
                layernorm=config['encoder_layernorm'],
                dropout=config['encoder_dropout'],
                trainable_initial=True,
                offset=0))
        self.add(Sequence(
            'decoder', LSTM, False,
            config['trg_embedding_dims'],
            config['decoder_state_dims'],
            layernorm=config['decoder_layernorm'],
            dropout=config['decoder_dropout'],
            attention_dims=config['attention_dims'],
            attended_dims=2*config['encoder_state_dims'],
            trainable_initial=False,
            offset=-1))

        h_t = T.matrix('h_t')
        self.predict_fun = function(
                [h_t],
                T.nnet.softmax(self.emission(T.tanh(self.hidden(h_t)))))

        inputs = T.lmatrix('inputs')
        inputs_mask = T.bmatrix('inputs_mask')
        self.encode_fun = function(
                [inputs, inputs_mask],
                self.encode(inputs, inputs_mask))

    def xent(self, inputs, inputs_mask, outputs, outputs_mask):
        pred_outputs, pred_attention = self(
                inputs, inputs_mask, outputs, outputs_mask)
        outputs_xent = batch_sequence_crossentropy(
                pred_outputs, outputs[1:], outputs_mask[1:])
        return outputs_xent

    def loss(self, *args):
        outputs_xent = self.xent(*args)
        return super().loss() + outputs_xent

    def search(self, inputs, inputs_mask, max_length):
        h_0, c_0, attended = self.encode_fun(inputs, inputs_mask)
        return self.decoder.search(
                self.predict_fun,
                self._trg_embeddings.get_value(borrow=True),
                self.config['trg_encoder']['<S>'],
                self.config['trg_encoder']['</S>'],
                max_length,
                states_0=[h_0, c_0],
                attended=attended,
                attention_mask=inputs_mask)

    def encode(self, inputs, inputs_mask):
        embedded_inputs = self._src_embeddings[inputs]
        # Forward encoding pass
        fwd_h_seq, fwd_c_seq = self.fwd_encoder(embedded_inputs, inputs_mask)
        # Backward encoding pass, using hidden states from forward encoder
        back_h_seq, back_c_seq = self.back_encoder(
                T.concatenate([embedded_inputs, fwd_h_seq], axis=-1),
                inputs_mask)
        # Initial states for decoder
        h_0 = back_h_seq[-1]
        c_0 = back_c_seq[-1]
        # Attention on concatenated forward/backward sequences
        attended = T.concatenate([fwd_h_seq, back_h_seq], axis=-1)
        return h_0, c_0, attended

    def __call__(self, inputs, inputs_mask, outputs, outputs_mask):
        embedded_outputs = self._trg_embeddings[outputs]
        h_0, c_0, attended = self.encode(inputs, inputs_mask)
        h_seq, c_seq, attention_seq = self.decoder(
                embedded_outputs, outputs_mask, states_0=(h_0, c_0),
                attended=attended, attention_mask=inputs_mask)
        pred_seq = softmax_3d(self.emission(T.tanh(self.hidden(h_seq))))

        return pred_seq, attention_seq


def main():
    import argparse
    import pickle
    import sys
    import os.path
    from time import time

    parser = argparse.ArgumentParser(
            description='Neural machine translation')

    parser.add_argument('--model', type=str, required=True,
            help='name of the model file')
    parser.add_argument('--corpus', type=str,
            help='name of parallel corpus file')

    args = parser.parse_args()

    if os.path.exists(args.model):
        with open(args.model, 'rb') as f:
            config = pickle.load(f)
            model = NMT('nmt', config)
            model.load(f)
    else:
        n_epochs = 1
        batch_size = 64
        test_size = batch_size
        max_length = 30

        with open(args.corpus, 'r', encoding='utf-8') as f:
            def read_pairs():
                for line in f:
                    fields = [s.strip() for s in line.split('|||')]
                    if len(fields) == 2:
                        pair = tuple(map(str.split, fields))
                        lens = tuple(map(len, pair))
                        if min(lens) >= 2 and max(lens) <= max_length:
                            yield pair
            src_sents, trg_sents = list(zip(*read_pairs()))
            src_encoder = TextEncoder(sequences=src_sents, max_vocab=10000)
            trg_encoder = TextEncoder(sequences=trg_sents, max_vocab=10000)
            sent_pairs = list(zip(src_sents, trg_sents))
            print('Read %d sentences, vocabulary size %d/%d' % (
                len(sent_pairs), len(src_encoder), len(trg_encoder)),
                flush=True)
            
        config = {
            'src_encoder': src_encoder,
            'trg_encoder': trg_encoder,
            'src_embedding_dims': 512,
            'trg_embedding_dims': 512,
            'encoder_dropout': 0.2,
            'decoder_dropout': 0.2,
            'encoder_state_dims': 1024,
            'decoder_state_dims': 1024,
            'attention_dims': 1024,
            'encoder_layernorm': 'ba1',
            'decoder_layernorm': 'ba1',
            }
        
        model = NMT('nmt', config)

        sym_inputs = T.lmatrix('inputs')
        sym_inputs_mask = T.bmatrix('inputs_mask')
        sym_outputs = T.lmatrix('outputs')
        sym_outputs_mask = T.bmatrix('outputs_mask')

        optimizer = Adam(
                model.parameters(),
                model.loss(sym_inputs, sym_inputs_mask,
                           sym_outputs, sym_outputs_mask),
                [sym_inputs, sym_inputs_mask],
                [sym_outputs, sym_outputs_mask],
                grad_max_norm=5.0)

        xent = function(
                [sym_inputs, sym_inputs_mask, sym_outputs, sym_outputs_mask],
                model.xent(sym_inputs, sym_inputs_mask,
                           sym_outputs, sym_outputs_mask))

        test_set = sent_pairs[:test_size]
        train_set = sent_pairs[test_size:]

        test_src, test_trg = list(zip(*test_set))
        test_inputs, test_inputs_mask = src_encoder.pad_sequences(test_src)
        test_outputs, test_outputs_mask = trg_encoder.pad_sequences(test_trg)

        start_time = time()
        end_time = start_time + 24*3600
        batch_nr = 0

        while time() < end_time:
            def pair_len(pair): return max(map(len, pair))
            for batch_pairs in iterate_batches(train_set, 64, pair_len):
                src_batch, trg_batch = list(zip(*batch_pairs))
                inputs, inputs_mask = src_encoder.pad_sequences(src_batch)
                outputs, outputs_mask = trg_encoder.pad_sequences(trg_batch)
                t0 = time()
                train_loss = optimizer.step(
                        inputs, inputs_mask, outputs, outputs_mask)
                print('Train loss: %.3f (%.2f s)' % (train_loss, time()-t0),
                      flush=True)
                batch_nr += 1
                if batch_nr % 10 == 0:
                    test_xent = xent(test_inputs, test_inputs_mask,
                                     test_outputs, test_outputs_mask)
                    print('Test xent: %.3f' % test_xent, flush=True)
                if batch_nr % 100 == 0:
                    pred, pred_mask, scores = model.search(
                            test_inputs, test_inputs_mask, max_length)
                    for src_sent, sent, sent_mask, score in zip(
                            test_inputs.T,
                            pred[-1].T, pred_mask[-1].T, scores[-1].T):
                        print(' '.join(
                            src_encoder.vocab[x] for x in src_sent.flatten()
                            if x > 1))
                        print('%.2f'%score, ' '.join(
                            trg_encoder.vocab[x] for x, there
                            in zip(sent.flatten(), sent_mask.flatten())
                            if bool(there)))
                        print('-'*72, flush=True)

                if time() >= end_time: break

        with open(args.model, 'wb') as f:
            pickle.dump(config, f)
            model.save(f)


if __name__ == '__main__': main()

