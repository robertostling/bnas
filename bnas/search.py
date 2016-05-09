"""Search algorithms for recurrent networks."""

import theano
import numpy as np

def greedy(step, states0, batch_size, start_symbol, stop_symbol, max_length):
    """Greedy search algorithm.

    Parameters
    ----------
    step : function
        Function which takes the following arguments:
        (iteration number, list of states, output history,
        output history mask)
        It is expected to return a tuple (new_states, outputs),
        where `new_states` is of the same form as `states0` and outputs
        is a matrix of shape (batch_size, n_symbols) containing probability
        distributions for the next symbol in each sequence of the batch.
    states0 : list
        Initial states, this will be passed to the `step` function on the
        first iteration.
    batch_size : int
        Batch size, this should be derivable from states0 but is included as
        an extra parameter for clarity.
    start_symbol : int
        Start symbol in the encoding used.
    stop_symbol : int
        Stop symbol in the encoding used.
    max_length : int
        Maximum length of sequence, but shorter sequences may also be
        returned.

    Returns
    -------
    r : tuple
        Tuple of (outputs, outputs_mask), numpy arrays of dtype int64 and
        theano.config.floatX respective.
    """

    # Output mask at each time step, first step (with start symbols) is
    # completely covered.
    output_mask_seq = [np.ones((batch_size,), dtype=theano.config.floatX)]
    # Output symbols over the batches, at each time step.
    output_seq = [np.full((batch_size,), start_symbol, dtype=np.int64)]

    states = states0
    for i in range(max_length-2):
        states, output_dists = step(i, states, output_seq, output_mask_seq)
        output_seq.append(output_dists.argmax(axis=-1))
        output_mask_seq.append(
                (output_seq[-2] != stop_symbol) * output_mask_seq[-1])
        active = (output_seq[-1] != stop_symbol) * output_mask_seq[-1]
        if not active.any():
            return np.array(output_seq), np.array(output_mask_seq)

    # Add stop symbols to any unterminated sequneces.
    output_seq.append(
            np.full((batch_size,), stop_symbol, dtype=np.int64))
    output_mask_seq.append(output_mask_seq[-1])

    return np.array(output_seq), np.array(output_mask_seq)

