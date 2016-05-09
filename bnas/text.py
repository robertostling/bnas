from collections import Counter

import numpy as np
import theano


def encode_sequences(sequences, max_n_symbols=None,
                     special=('<S>', '</S>', '<UNK>'), dtype=np.int64):

    if (not max_n_symbols is None) and ('<UNK>' not in special):
        raise ValueError('Missing <UNK> symbol with limited vocabulary')

    def create_vocab():
        counts = Counter(sym for seq in sequences for sym in seq)
        if max_n_symbols is None:
            vocab = special + tuple(sorted(counts.keys()))
        else:
            vocab = special + tuple(sorted(
                    sym for sym, _ in
                        counts.most_common(max_n_symbols-len(special))))
        return vocab

    vocab = create_vocab()
    index = {sym:i for i,sym in enumerate(vocab)}
    unk = None if max_n_symbols is None else index['<UNK>']
    prefix = [index['<S>']] if '<S>' in special else []
    suffix = [index['</S>']] if '</S>' in special else []

    encoded = [
        np.array(prefix+[index.get(sym, unk) for sym in seq]+suffix,
                 dtype=dtype)
        for seq in sequences]

    return vocab, index, encoded


def mask_sequences(encoded, max_length=None, dtype=theano.config.floatX):
    length = max(map(len, encoded))
    if not max_length is None:
        length = min(length, max_length)
    batch_size = len(encoded)
    mask = np.array([
        [i < len(seq) for i in range(length)]
        for seq in encoded],
        dtype=dtype)
    def adjust(seq):
        if len(seq) > length: return seq[:length]
        return np.concatenate([seq, np.ones(length-len(seq), dtype=seq.dtype)])
    batch = np.array([adjust(seq) for seq in encoded])
    return batch, mask

