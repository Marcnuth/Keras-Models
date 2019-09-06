from kerasmodels.models.skip_gram import SkipGramDataGenerator, SkipGram
from pathlib import Path


def test_data_gen():
    gen = SkipGramDataGenerator(datafile=Path(__file__).resolve().parent / 'text.txt', batch_size=32)
    print(gen.summary())

    for i in range(88):
        print(i, gen[i][0].shape)
