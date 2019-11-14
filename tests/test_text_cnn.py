from keras_models.models import TextCNN
from keras.optimizers import Adadelta, SGD
from keras.losses import mean_squared_error
import numpy as np
from keras_models.generators.allonce import AllOnceFromDirGenerator
from pathlib import Path
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import time
from keras.callbacks import TensorBoard, EarlyStopping


def test_train():

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    set_session(sess)

    timestamp = str(int(time.time()))
    tb_dir = Path('~/xx/tensorboard/')

    callbacks = [
        TensorBoard(log_dir=tb_dir.absolute().as_posix(), histogram_freq=0, write_graph=True, write_images=True),
        # EarlyStopping(monitor='categorical_accuracy', min_delta=1e-5, patience=5, verbose=2, mode='auto')
    ]

    datagen = AllOnceFromDirGenerator(
        datadir=Path('/home/ubuntu/xx/aiflow/tests/resources/titles/seniority_data/'),
        batch_size=64, shuffle=True, random_seed=1, limit=100000
    )
    datagen.summary()

    model = TextCNN(input_shape=(5, 300), n_classes=4, filter_sizes=(3, 4, 5))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.summary()

    model.fit_generator(
        datagen,
        steps_per_epoch=datagen.steps_each_epoch,
        epochs=200,
        verbose=2,
        workers=10,
        max_queue_size=32,
        callbacks=callbacks
    )

    print('finish training, save the model')
    model.save((Path(__file__).parent / 'output' / 'textcnn_seniroty.h5').absolute().as_posix())
