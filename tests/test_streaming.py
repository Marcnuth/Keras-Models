from keras_models.models import TextCNN
from keras.optimizers import Adadelta, SGD
from keras.losses import mean_squared_error
import numpy as np
from keras_models.generators.streaming import StreamingFromDirGenerator
from pathlib import Path
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import time
from keras.callbacks import TensorBoard, EarlyStopping


def test_dataset():

    datagen = StreamingFromDirGenerator(
        datadir=Path('/home/ubuntu/xx/aiflow/tests/resources/titles/seniority_data/'),
        batch_size=64, shuffle=False, random_seed=1
    )

    print(datagen[1])
    print(datagen.summary())
