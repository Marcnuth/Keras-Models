from kerasmodels.models import DNN
from keras.optimizers import Adadelta, SGD
from keras.losses import mean_squared_error
import numpy as np


def test_linear():
    X = np.random.normal(0, 1.0, size=5000).reshape(500, 10)
    w = np.random.normal(0, 1.0, size=10)
    Y = np.dot(X, w) + np.random.randint(1)

    model = DNN(input_shape=X.shape[1:], output_shape=1, output_activation="sigmoid", dtype=float, r_drop=0.5, dense_layers=[512, 128])
    model.compile(optimizer='adam', loss=mean_squared_error, metrics=['mae', 'mse'])
    model.summary()

    model.fit(X, Y, batch_size=16, epochs=100, validation_split=0.1)

    print(X.shape, w.shape, Y.shape)