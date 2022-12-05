# Image Classification and Transfer Learning with LeNet-5

## 1. Introduction
We perform image classfication on the CIFAR-100 dataset using the [Lenet-5](https://ieeexplore.ieee.org/document/726791), a type of Convolutional Neural Network (CNN). We first train and test the model on the CIFAR-100 dataset. The [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) dataset consists of 60000 32x32 colour images in 100 classes containing 600 images each. There are 500 training images and 100 testing images per class. We then perform transfer learning by fine-tuning the Lenet-5 model to classify images on the STL-10 dataset. The [STL-10](https://cs.stanford.edu/~acoates/stl10/) dataset consists of 5000 96x96 colour images in 10 classes containing 500 images each. There are 400 training images and 100 testing images per class.

## 1. Image Classification on CIFAR-100

## 1.1 Vanilla Lenet-5
We create custom dataloaders for the CIFAR-100 dataset. We then build the LeNet-5 model with the PyTorch Sequential module. We train the model for 10 epochs and use a learning rate of 0.001 and batch size of 64. We use the Adam optimizer for training. We use the cross-entropy loss function. We save the model with the best validation accuracy. We then evaluate the model on the test set.

## 1.2 Architecture & Hyperparameter Tuning
We change the architecture of Le-Net5 as follows:
- change activation function from tanh to ReLU
- change pooling layers from average pooling to max pooling
- add a batch normalization layer after the first fully connected layer
- use hidden fc layers of dims 64, 128

We re-train the model with the new architecture and hyperparameters. We use the same learning rate of 0.001 and batch size of 64, and train for 20 epochs. We then evaluate the model on the test set and compare the results with the vanilla LeNet-5 model.

## 2. Transfer Learning on STL-10
We initialize the LeNet-5 model using the fine-tuned architecture from section [1.2](#12-architecture--hyperparameter-tuning). We freeze all layers except the last fully connected layer. We then train the model for 10 epochs and use a learning rate of 0.001 and batch size of 64. We use the Adam optimizer for training. We use the cross-entropy loss function. We save the model with the best validation accuracy. We then evaluate the model on the test set.