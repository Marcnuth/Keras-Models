# Keras Models Hub

![PyPI - Downloads](https://img.shields.io/pypi/dm/keras-models?label=PyPI)

This repo aims at providing both **reusable** Keras Models and **pre-trained** models, which could easily integrated into your projects.

## Install

```shell
pip install keras-models
```

If you will using the NLP models, you need run one more command:
```shell
python -m spacy download xx_ent_wiki_sm
```

## Usage Guide

```
import kearasmodels

```


## Examples

### Reusable Models

__SkipGram__

__WideDeep__

### Pre-trained Models

__VGG16_Places365__
> This model is forked from [GKalliatakis/Keras-VGG16-places365](https://github.com/GKalliatakis/Keras-VGG16-places365) and [CSAILVision/places365](https://github.com/CSAILVision/places365)

```
from kerasmodels.models.pretrained import vgg16_places365
labels = vgg16_places365.predict('your_image_file_pathname.jpg', n_top=3)
# Example Result: labels = ['cafeteria', 'food_court', 'restaurant_patio'] 
```


## Models

- TextCNN
- TextDNN
- SkipGram
- VGG16_Places365 [pre-trained]
- working on more models

## Citation

__TextCNN__

```
Kim Y. 
Convolutional neural networks for sentence classification[J]. 
arXiv preprint arXiv:1408.5882, 2014.
```

__SkipGram__

```
Mikolov T, Chen K, Corrado G, et al. 
Efficient estimation of word representations in vector space[J]. 
arXiv preprint arXiv:1301.3781, 2013.
```


__VGG16_Places365__
```
Zhou, B., Lapedriza, A., Khosla, A., Oliva, A., & Torralba, A.
Places: A 10 million Image Database for Scene Recognition
IEEE Transactions on Pattern Analysis and Machine Intelligence
```

## Contribution

Please submit PR if you want to contribute, or submit issues for new model requirements.

