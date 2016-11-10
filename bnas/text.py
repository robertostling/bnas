"""Text processing.

The :class:`TextEncoder` class is the main feature of this module, the helper
functions were used in earlier examples and should be phased out.
"""

from collections import Counter

import numpy as np
import theano


class TextEncoder:
    def __init__(self,
                 max_vocab=None,
                 min_count=None,
                 vocab=None,
                 sequences=None,
                 sub_encoder=None,
                 special=('<S>', '</S>', '<UNK>')):
        self.sub_encoder = sub_encoder
        self.special = special

        if vocab is not None:
            self.vocab = vocab
        else:
            if sequences is not None:
                c = Counter(x for xs in sequences for x in xs)
                if max_vocab is not None:
                    vocab_sorted = sorted(c.items(),
                                          key=lambda t: (-t[1], t[0]))
                    self.vocab = special + tuple(sorted(
                        s for s,_ in vocab_sorted[:max_vocab]))
                elif min_count is not None:
                    self.vocab = special + tuple(sorted(
                            s for s,n in c.items() if n >= min_count))
                else:
                    self.vocab = special + tuple(sorted(c.keys()))

        self.index = {s:i for i,s in enumerate(self.vocab)}

    def __str__(self):
        if self.sub_encoder is None:
            return 'TextEncoder(%d)' % len(self)
        else:
            return 'TextEncoder(%d, %s)' % (len(self), str(self.sub_encoder))

    def __repr__(self):
        return str(self)

    def __getitem__(self, x):
        return self.index.get(x, self.index.get('<UNK>'))

    def __len__(self):
        return len(self.vocab)

    def encode_sequence(self, sequence, max_length=None, unknowns=None):
        start = (self.index['<S>'],) if '<S>' in self.index else ()
        stop = (self.index['</S>'],) if '</S>' in self.index else ()
        unk = self.index.get('<UNK>')
        def encode_item(x):
            idx = self.index.get(x)
            if idx is None:
                if unknowns is None:
                    return unk
                else:
                    unknowns.append(x)
                    return -len(unknowns)
            else:
                return idx
        encoded = tuple(idx for idx in list(map(encode_item, sequence))
                        if idx is not None)
        if max_length is None \
        or len(encoded)+len(start)+len(stop) <= max_length:
            return start + encoded + stop
        else:
            return start + encoded[:max_length-(len(start)+len(stop))] + stop

    def pad_sequences(self, sequences,
                      max_length=None, pad_right=True, dtype=np.int32):
        if not sequences:
            # An empty matrix would mess up things, so create a dummy 1x1
            # matrix with an empty mask in case the sequence list is empty.
            m = np.zeros((1 if max_length is None else max_length, 1),
                         dtype=dtype)
            mask = np.zeros_like(m, dtype=np.bool)
            return m, mask
        unknowns = None if self.sub_encoder is None else []
        encoded_sequences = [
                self.encode_sequence(sequence, max_length, unknowns)
                for sequence in sequences]
        length = max(map(len, encoded_sequences))
        length = length if max_length is None else min(length, max_length)

        m = np.zeros((length, len(sequences)), dtype=dtype)
        mask = np.zeros_like(m, dtype=np.bool)

        for i,encoded in enumerate(encoded_sequences):
            if pad_right:
                m[:len(encoded),i] = encoded
                mask[:len(encoded),i] = 1
            else:
                m[-len(encoded):,i] = encoded
                mask[-len(encoded):,i] = 1

        if unknowns is None:
            return m, mask
        else:
            char, char_mask = self.sub_encoder.pad_sequences(unknowns)
            return m, mask, char, char_mask

    def decode_padded(self, m, mask, char=None, char_mask=None):
        if char is not None:
            unknowns = list(map(
                ''.join, self.sub_encoder.decode_padded(char, char_mask)))
        start = self.index.get('<S>')
        stop = self.index.get('</S>')
        return [[unknowns[-x-1] if x < 0 else self.vocab[x]
                 for x,b in zip(row,row_mask)
                 if bool(b) and x not in (start, stop)]
                for row,row_mask in zip(m.T,mask.T)]


def encode_sequences(sequences, max_n_symbols=None, hybrid=False,
                     special=('<S>', '</S>', '<UNK>'), dtype=np.int64):
    """Encode sequences as numpy arrays of integers.

    Parameters
    ----------
    sequences : list
        List of sequences to encode.
    max_n_symbols : int, optional
        If given, the total number of symbols (including special symbols) is
        limited to this number. The most common symbols in the given sequences
        are used.
    hydrid : bool
        If False, encode all words except the max_n_symbols most common as
        <UNK>, otherwise ...
    special : tuple of str
        List of special symbols to include.
    dtype : numpy dtype
        Datatype of returned arrays.

    Returns
    -------
    vocabulary : list
        List of vocabulary items, in order.
    index : dict
        ``{v:i for i,v in enumerate(vocabulary)}``
    encoded_sequences : list
        List of numpy arrays with the encoded `sequences`.
    """

    if (not max_n_symbols is None) and ('<UNK>' not in special):
        raise ValueError('Missing <UNK> symbol with limited vocabulary')

    def create_vocab():
        counts = Counter(sym for seq in sequences for sym in seq)
        n_symbols = None if max_n_symbols is None \
                    else max_n_symbols - len(special)
        vocab = special + tuple(sorted(
                sym for sym, _ in
                    counts.most_common(n_symbols)))
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


def mask_sequences(encoded, max_length=None, dtype=np.int8):
    """Create a masked matrix of sequences.

    Parameters
    ----------
    encoded : list
        List of numpy arrays, possibly from :func:`encode_sequences`.
    max_length : int
        If given, this is the maximum number of columns in the returned
        matrices.
    dtype : numpy dtype
        Datatype of the `mask` return value.

    Returns
    -------
    matrix : numpy array
        Single array containing all the sequences in `encoded`.
        This array has shape (sequence_length, batch_size), which is the
        transpose of what is obtained by padding and concatenating the inputs.
    mask : numpy array (int8)
        Mask for `matrix`.
    """

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
    # TODO: any later performance advantage from physically transposing the
    # data?
    return batch.T, mask.T

