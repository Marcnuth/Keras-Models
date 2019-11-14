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


class StreamingFromDirGenerator(Sequence):

    def __init__(self, datadir, batch_size, r_sample=0.001, random_seed=None, shuffle=True):

        self.datadir = Path(datadir)
        self.instances = [f for f in self.datadir.iterdir() if f.is_file() and f.name.endswith('npz')]
        assert self.datadir.exists() and self.datadir.is_dir(), f'path:{datadir} is not a folder or does not exist.'

        self.shuffle = shuffle
        self.r_sample = r_sample
        self.batch_size = batch_size

        np.random.seed(random_seed or int(time()))
        self.on_epoch_end()

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):

        instances = self.instances[index * self.batch_size, (index + 1) * self.batch_size]

        X, Y = [], []
        for i, ins in enumerate(instances):
            with np.load(ins.absolute().as_posix()) as data:
                X.append(data['x'])
                Y.append(data['y'])

        X = np.concatenate(X).reshape(len(X), -1)
        Y = np.concatenate(Y).reshape(len(Y), -1)

        return X, Y

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.instances)
        return super().on_epoch_end()
