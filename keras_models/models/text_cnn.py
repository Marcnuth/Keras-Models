from keras.layers import Input, Embedding, Concatenate, Reshape, Dropout, Dense, MaxPool2D, Conv2D
from keras.models import Model
from keras.initializers import TruncatedNormal, constant
from keras.backend import expand_dims
from keras.utils import Sequence


def build(input_shape, n_classes, filter_sizes, **kwargs):
    '''
    filter_sizes: a list. each element is the filter for CONV
    '''

    n_filters = kwargs.get('n_filters', 128)
    r_dropout = kwargs.get('r_dropout', 0.5)

    n_text_len, n_embedding_len = input_shape

    x = Input(shape=input_shape, name='input_x')
    expand_x = expand_dims(x, axis=-1)

    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        conv = Conv2D(
            filters=n_filters, kernel_size=[filter_size, n_embedding_len], strides=1, padding='valid', activation='relu',
            kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.1), bias_initializer=constant(value=0.1), name=('conv_%d' % filter_size)
        )(expand_x)
        pool = MaxPool2D(pool_size=[n_text_len - filter_size + 1, 1], strides=(1, 1), padding='valid', name=('pool_%d' % filter_size))(conv)
        pooled_outputs.append(pool)

    # combine all the pooled features
    n_filters_total = n_filters * len(filter_sizes)
    h_pool = Concatenate(axis=3)(pooled_outputs)
    h_pool_flat = Reshape([n_filters_total])(h_pool)
    dropout = Dropout(r_dropout)(h_pool_flat)

    output = Dense(n_classes, kernel_initializer='glorot_normal',  bias_initializer=constant(0.1), activation='softmax', name='output')(dropout)
    model = Model(inputs=x, outputs=output)

    return model
