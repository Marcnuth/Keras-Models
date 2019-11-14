from keras_models.models import LinearModel
from keras.optimizers import Adadelta, SGD
from keras.losses import mean_squared_error
import numpy as np


def test_linear():
    X = np.random.normal(0, 1.0, size=5000).reshape(500, 10)
    w = np.random.normal(0, 1.0, size=10)
    Y = np.dot(X, w) + np.random.randint(1)

    model = LinearModel(input_shape=X.shape[1:], output_shape=1, dtype=float)
    model.compile(optimizer=SGD(), loss=mean_squared_error, metrics=['mae', 'mse'])
    model.summary()

    model.fit(X, Y, batch_size=16, epochs=10, validation_split=0.1)

    print(X.shape, w.shape, Y.shape)
