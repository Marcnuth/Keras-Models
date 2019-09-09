from keras_models.models import CNN
from keras.optimizers import Adadelta, SGD
from keras.losses import mean_squared_error
import numpy as np


def test_linear():
    X = np.random.normal(0, 1.0, size=500 * 100 * 100 * 3).reshape(500, 100, 100, 3)
    w1 = np.random.normal(0, 1.0, size=100)
    w2 = np.random.normal(0, 1.0, size=3)
    Y = np.dot(np.dot(np.dot(X, w2), w1), w1) + np.random.randint(1)

    print(X.shape, Y.shape)

    model = CNN(input_shape=X.shape[1:], filters=[32, 64], kernel_size=(2, 2), pool_size=(3, 3), padding='same', r_dropout=0.25, num_classes=1)
    model.compile(optimizer='adam', loss=mean_squared_error, metrics=['mae', 'mse'])
    model.summary()

    model.fit(X, Y, batch_size=16, epochs=100, validation_split=0.1)

    print(X.shape, Y.shape)
