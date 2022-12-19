# Final Project for CIS5810: Neural Style Transfer in JAX

## Introduction
In this work we experiment with multiple different strategies for neural based style transfer on both videos and images. We follow previously published methodology proposed by Gatys et.al while attempting to optimize the output, trying a variety of different networks for feature extraction, different optimizers different loss functions and different regularization terms to produce the ideal output. In the end, we try to extend our work into video style transfer.

## Requirement
We write the main code on Google Colab and include the command line to install these necessary packages.
Packages used in this project:
- [Google JAX](https://github.com/google/jax)
- [Flax](https://github.com/google/flax): A neural network library and ecosystem for JAX designed for flexibility
- [Flaxmodels](https://github.com/matthias-wright/flaxmodels): A collection of pre-trained models in Flax, by Matthias-wright
- [Optax](https://github.com/deepmind/optax): A gradient processing and optimization library for JAX, provide optimizers and Huber loss
- [ClosedFormMatting](https://github.com/MarcoForte/closed-form-matting): A package used for calculating the Matting Laplacian

