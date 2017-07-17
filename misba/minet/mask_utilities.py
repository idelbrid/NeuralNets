# Borrowed thankfully from sergeyf at https://github.com/fchollet/keras/issues/1579

import keras.backend as K
from keras.engine.topology import Layer


class ZeroMaskedEntries(Layer):
    """
    This layer is called after an Embedding layer.
    It zeros out all of the masked-out embeddings.
    It also swallows the mask without passing it on.
    You can change this to default pass-on behavior as follows:

    def compute_mask(self, x, mask=None):
        if not self.mask_zero:
            return None
        else:
            return K.not_equal(x, 0)
    """

    def __init__(self, **kwargs):
        self.support_mask = True
        super(ZeroMaskedEntries, self).__init__(**kwargs)

    def build(self, input_shape):
        self.output_dim = input_shape[1]
        self.repeat_dim = input_shape[2]

    def call(self, x, mask=None):
        mask = K.cast(mask, 'float32')
        mask = K.repeat(mask, self.repeat_dim)
        mask = K.permute_dimensions(mask, (0, 2, 1))
        return x * mask

    def compute_mask(self, input_shape, input_mask=None):
        return None


def mask_aware_mean(x):
    # recreate the masks - all zero rows have been masked
    mask = K.not_equal(K.sum(K.abs(x), axis=2, keepdims=True), 0)

    # number of that rows are not all zeros
    n = K.sum(K.cast(mask, 'float32'), axis=1, keepdims=False)

    # compute mask-aware mean of x
    x_mean = K.sum(x, axis=1, keepdims=False) / n

    return x_mean


def mask_aware_mean_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 3
    return (shape[0], shape[2])


def mask_aware_max(x):
    mask = K.not_equal(K.sum(K.abs(x), axis=2, keepdims=True), 0)
    mask = K.cast(mask, 'float32')
    vecmin = K.min(x, axis=1, keepdims=True)

    xstar = x + (vecmin * (1 - mask))  # setting masked values to the min value

    return K.max(xstar, axis=1, keepdims=False)


def logsumexp(x):
    return K.logsumexp(x, axis=1, keepdims=False)


if __name__ == '__main__':
    import numpy as np
    from keras.layers import Input, Lambda, Masking
    from keras.models import Model

    output_dim = 2
    n_instances = 4
    n_features = 2
    main_input = Input(shape=(n_instances, n_features), dtype='float32')

    masked_input = Masking()(main_input)
    input_zeroed = ZeroMaskedEntries()(masked_input)
    lambda_mean = Lambda(mask_aware_mean, mask_aware_mean_output_shape)(input_zeroed)
    lambda_max = Lambda(mask_aware_max, mask_aware_mean_output_shape)(input_zeroed)

    model = Model(inputs=main_input, outputs=lambda_mean)
    model.compile(optimizer='rmsprop', loss='mse')

    # test
    test_input = [[[1, 1], [2, 2], [0, 0], [0, 0]],
                  [[2, 2], [3, 3], [1, 1], [0, 0]],
                  [[0, 0], [0, 0], [0, 0], [1, 2]]]
    test_output = model.predict(test_input)
    print(test_output)

    max_model = Model(inputs=main_input, outputs=lambda_max)
    max_model.compile(optimizer='rmsprop', loss='mse')

    test_output = max_model.predict(test_input)
    print(test_output)



    # print('Mean is working?', np.all(np.isclose(test_output[0:2, :].mean(0), test_output[2, :])))