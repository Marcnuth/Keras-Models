from keras.layers import Dense, Input
from keras.models import Model
from pathlib import Path
import spacy
from itertools import product, chain
from keras.utils import Sequence
from collections import Counter
import numpy as np
from time import time
from keras import backend as K
import random


class AllOnceFromDirGenerator(Sequence):

    def __init__(self, datadir, batch_size, random_seed=None, shuffle=True, limit=None):

        self.datadir = Path(datadir)
        self.files = [f for f in self.datadir.iterdir() if f.is_file() and f.name.endswith('npz')]
        assert self.datadir.exists() and self.datadir.is_dir(), f'path:{datadir} is not a folder or does not exist.'

        if shuffle:
            random.shuffle(self.files)
        if limit:
            self.files = self.files[:limit]

        self.shuffle = shuffle
        self.batch_size = batch_size

        self.__load_all_instances()
        self.steps_each_epoch = self.X.shape[0] // self.batch_size + 1

        np.random.seed(random_seed or int(time()))
        self.on_epoch_end()

    def __load_all_instances(self):
        X, Y = [], []

        for f in self.files:
            with np.load(f.absolute().as_posix()) as data:
                X.append(data['x'])
                Y.append(data['y'])

        self.X = np.array(X)
        self.Y = np.array(Y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):

        istart = index * self.batch_size % self.X.shape[0]
        return self.X[istart:istart + self.batch_size], self.Y[istart:istart + self.batch_size]

    def summary(self, stdout=True):
        msg = f'''>>> Streaming Data Generator Summary:
        | Name             |                       Value                        |
        |------------------+----------------------------------------------------|
        | data_dir         | {self.datadir.absolute().as_posix():^50s} |
        | batch_size       | {self.batch_size:^50d} |
        | X shape          | {str(self.X.shape):^50s} |
        | Y shape          | {str(self.Y.shape):^50s} |
        | steps_each_epoch | {self.steps_each_epoch:^50d} |
        \n'''

        if stdout:
            print(msg)
        return msg
