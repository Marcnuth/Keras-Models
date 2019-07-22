from . import LinearModel, DNN
from keras.layers import Input, Dense, Dropout
from keras.models import Model
import logging


logger = logging.getLogger(__name__)


def build(input_shape, output_shape, dtype, dense_layers=[1024, 512, 128], r_drop=0.5, name="wide_deep"):
    assert len(dense_layers) >= 2, 'invalid dense layers'

    x = Input(shape=input_shape, dtype=dtype, name="deep_input")
    x = Dense(dense_layers[0], activation='relu', name="deep_dense_1")(x)
    x = Dropout(r_drop, name="deep_dropout_1")(x)
    x = Dense(dense_layers[1], activation='relu', name="deep_dense_2")(x)
    x = Dropout(r_drop, name="deep_dropout_2")(x)

    x = Dense(dense_layers[1], activation='relu', name="deep_dense_2")(x)


