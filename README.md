# Deep-Learning-for-Fashion-MNIST-Classification

## Overview
A simple deep learning model for predicting labels of handwritten images from the Fashion-MNIST dataset, it must be run in a Linux environment.

## Data: FashionMNIST ([source](https://github.com/zalandoresearch/fashion-mnist))

### General

<b>Fashion-MNIST</b> is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes.

![fashion-mnist-sprite](https://user-images.githubusercontent.com/72423203/191127147-7b917365-f512-4bb2-9af6-529b10d49e23.png)

It typically serves as a replacement for the original MNIST dataset containing hand-drawn digits, which is widely regarded for benchmarking machine learning algorithms.

### Classes

Each image in the dataset is associated with a label from 1 of 10 classes (given below)

<table>
  <tr> 
    <th>Label</th>	
    <th>Description</th>
  </tr>
   <tr> 
     <td>0/t</td>
     <td>T-shirt/t</td>
  </tr>
0	T-shirt/top
1	Trouser
2	Pullover
3	Dress
4	Coat
5	Sandal
6	Shirt
7	Sneaker
8	Bag
9	Ankle boot
</table>

## Functions

For this program I have included 5 functions (listed below) to build a neural network, train the network, evaluate its performance, and make preditions on test data. 
