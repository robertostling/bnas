"""Text processing."""

from collections import Counter

import numpy as np
import theano


def encode_sequences(sequences, max_n_symbols=None,
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

