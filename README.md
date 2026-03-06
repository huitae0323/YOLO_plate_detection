# Vision Transformer and MLP-Mixer Architectures

In this repository we release models from the papers

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601)
- [How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers](https://arxiv.org/abs/2106.10270)
- [When Vision Transformers Outperform ResNets without Pretraining or Strong Data Augmentations](https://arxiv.org/abs/2106.01548)
- [LiT: Zero-Shot Transfer with Locked-image text Tuning](https://arxiv.org/abs/2111.07991)
- [Surrogate Gap Minimization Improves Sharpness-Aware Training](https://arxiv.org/abs/2203.08065)

The models were pre-trained on the [ImageNet](http://www.image-net.org/) and
[ImageNet-21k](http://www.image-net.org/) datasets. We provide the code for
fine-tuning the released models in
[JAX](https://jax.readthedocs.io)/[Flax](http://flax.readthedocs.io).

The models from this codebase were originally trained in
https://github.com/google-research/big_vision/
where you can find more advanced code (e.g. multi-host training), as well as
some of the original training scripts (e.g.
[configs/vit_i21k.py](https://github.com/google-research/big_vision/blob/main/big_vision/configs/vit_i21k.py)
for pre-training a ViT, or
[configs/transfer.py](https://github.com/google-research/big_vision/blob/main/big_vision/configs/transfer.py)
for transfering a model).

Table of contents:

- [Vision Transformer and MLP-Mixer Architectures](#vision-transformer-and-mlp-mixer-architectures)
	- [Colab](#colab)
	- [Installation](#installation)
	- [Fine-tuning a model](#fine-tuning-a-model)
	- [Vision Transformer](#vision-transformer)
		- [Available ViT models](#available-vit-models)
		- [Expected ViT results](#expected-vit-results)
	- [MLP-Mixer](#mlp-mixer)
		- [Available Mixer models](#available-mixer-models)
		- [Expected Mixer results](#expected-mixer-results)
	- [LiT models](#lit-models)
	- [Running on cloud](#running-on-cloud)
		- [Create a VM](#create-a-vm)
		- [Setup VM](#setup-vm)
	- [Bibtex](#bibtex)
	- [Disclaimers](#disclaimers)
	- [Changelog](#changelog)


## Colab

Below Colabs run both with GPUs, and TPUs (8 cores, data parallelism).

The first Colab demonstrates the JAX code of Vision Transformers and MLP Mixers.
This Colab allows you to edit the files from the repository directly in the
Colab UI and has annotated Colab cells that walk you through the code step by
step, and lets you interact with the data.

https://colab.research.google.com/github/google-research/vision_transformer/blob/main/vit_jax.ipynb

The second Colab allows you to explore the >50k Vision Transformer and hybrid
checkpoints that were used to generate the data of the third paper "How to train
your ViT? ...". The Colab includes code to explore and select checkpoints, and
to do inference both using the JAX code from this repo, and also using the
popular [`timm`] PyTorch library that can directly load these checkpoints as
well. Note that a handful of models are also available directly from TF-Hub:
[sayakpaul/collections/vision_transformer] (external contribution by [Sayak
Paul]).
