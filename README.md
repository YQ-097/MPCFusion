# MPCFusion: Multiscale parallel cross-fusion for infrared and visible images via convolution and vision Transformer
Haojie Tang, Yao Qian, Mengliang Xing, Yisheng Cao, Gang Liu∗

Published in: Optics and Lasers in Engineering

- [paper](https://www.sciencedirect.com/science/article/abs/pii/S0143816624000745)

## Abstract
The image fusion community is thriving with the wave of deep learning, and the most popular fusion methods are usually built  upon well-designed network structures.  However, most of the current methods do not fully exploit deeper features while ignore the  importance of long-range dependencies.  In this paper, a convolution and vision Transformer-based multi-scale parallel cross fusion  network for infrared and visible images is proposed (MPCFusion).  To exploit deeper texture details, a feature extraction module  based on convolution and vision Transformer is designed.  With a view to correlating the shallow features between different modalities, a parallel cross-attention module is proposed, in which a parallel-channel model efficiently preserves the proprietary modal  features, followed by a cross-spatial model that ensures the information interactions between the different modalities.  Moreover,  a cross-domain attention module based on convolution and vision Transformer is proposed to capturing long-range dependencies  between in-depth features and effectively solves the problem of global context loss.  Finally, a nest-connection based decoder is used  for implementing feature reconstruction.  In particular, we design a new texture-guided structural similarity loss function to drive  the network to preserve more complete texture details.  Extensive experimental results illustrate that MPCFusion shows excellent  fusion performance and generalization capabilities.
## Framework
![image](https://github.com/YQ-097/MPCFusion/assets/68978140/fd1b9344-5fac-41d6-a714-17cfee1a870a)

## Recommended Environment

 - [x] torch 1.11.0
 - [x] torchvision 0.12.0
 - [x] tensorboard  2.7.0
 - [x] numpy 1.21.2

## To Train
Please modify the dataset path in the train_mpcfusion.json file, and then run the following file to start training.

    python main_train_mpcfusion.py
## To Test

    python test_mpcfusion.py
## Citation

```
@article{TANG2024108094,
title = {MPCFusion: Multi-scale parallel cross fusion for infrared and visible images via convolution and vision Transformer},
journal = {Optics and Lasers in Engineering},
volume = {176},
pages = {108094},
year = {2024},
issn = {0143-8166},
doi = {https://doi.org/10.1016/j.optlaseng.2024.108094},
url = {https://www.sciencedirect.com/science/article/pii/S0143816624000745},
author = {Haojie Tang and Yao Qian and Mengliang Xing and Yisheng Cao and Gang Liu},
keywords = {Image fusion, Vision Transformer, Convolution, Multi-scale feature, Infrared},
abstract = {The image fusion community is thriving with the wave of deep learning, and the most popular fusion methods are usually built upon well-designed network structures. However, most of the current methods do not fully exploit deeper features while ignore the importance of long-range dependencies. In this paper, a convolution and vision Transformer-based multi-scale parallel cross fusion network for infrared and visible images is proposed (MPCFusion). To exploit deeper texture details, a feature extraction module based on convolution and vision Transformer is designed. With a view to correlating the shallow features between different modalities, a parallel cross-attention module is proposed, in which a parallel-channel model efficiently preserves the proprietary modal features, followed by a cross-spatial model that ensures the information interactions between the different modalities. Moreover, a cross-domain attention module based on convolution and vision Transformer is proposed to capturing long-range dependencies between in-depth features and effectively solves the problem of global context loss. Finally, a nest-connection based decoder is used for implementing feature reconstruction. In particular, we design a new texture-guided structural similarity loss function to drive the network to preserve more complete texture details. Extensive experimental results illustrate that MPCFusion shows excellent fusion performance and generalization capabilities. The source code will be released at https://github.com/YQ-097/MPCFusion.}
}
```
