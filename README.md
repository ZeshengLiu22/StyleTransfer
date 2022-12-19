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
- [OpenCV](https://github.com/opencv/opencv): A package used for image processing and video processing
- [PIL](https://github.com/python-pillow/Pillow): A package for image processing

## Code files:
- Style Transfer V1.0: Baseline file
- Style Transfer V2.0: Clean the code file to make it more readable, rewrite several parts using class
- Style Transfer V3.0: Change the implementation of the pretrained model to ResNet models.
- Style Transfer V4.0: Add total variation regularization and photorealistic regularization
- Style Transfer V5.0: Change for different Loss
- Style Transfer V6.0: Video style transfer

## Results:
![Output1](https://github.com/ZeshengLiu22/StyleTransfer/blob/main/Image%20Style%20Transfer%20Results/FinalOutput1.jpg)
![Output2](https://github.com/ZeshengLiu22/StyleTransfer/blob/main/Image%20Style%20Transfer%20Results/FinalOutput2.jpg)
![Output3](https://github.com/ZeshengLiu22/StyleTransfer/blob/main/Image%20Style%20Transfer%20Results/FinalOutput3.jpg)

For output 2, we also use the method of image stitching to generate large output.
