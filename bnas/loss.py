import theano.tensor as T

def batch_sequence_crossentropy(x, target, target_mask):
    batch_size = x.shape[1]
    return (T.categorical_crossentropy(
            x.reshape(x.shape[0]*x.shape[1], x.shape[2]),
            target.flatten()) * target_mask.flatten()).sum() / batch_size

