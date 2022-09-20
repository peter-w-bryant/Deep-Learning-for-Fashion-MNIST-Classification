# Deep-Learning-for-Fashion-MNIST-Classification

## Overview
A simple deep learning model for predicting labels of greyscale images from the Fashion-MNIST dataset, it must be run in a Linux environment.

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
     <td>0</td>
     <td>T-shirt</td>
  </tr>
  <tr> 
     <td>1</td>
     <td>Trouser</td>
  </tr>
  <tr> 
     <td>2</td>
     <td>Pullover</td>
  </tr>
  <tr> 
     <td>3</td>
     <td>Dress</td>
  </tr>
  <tr> 
     <td>4</td>
     <td>Coat</td>
  </tr>
  <tr> 
     <td>5</td>
     <td>Sandal</td>
  </tr>
  <tr> 
     <td>6</td>
     <td>Shirt</td>
  </tr>
  <tr> 
     <td>7</td>
     <td>Sneaker</td>
  </tr>
   <tr> 
     <td>8</td>
     <td>Bag</td>
  </tr>
  <tr> 
     <td>9</td>
     <td>Boots</td>
  </tr>
</table>

## Functions

For this program I have included 5 functions (listed below) to build a neural network, train the network, evaluate its performance, and make preditions on test data. 

<ul>
  <li>
    <b>get_data_loader(training=True)</b>: A function to create a Dataloader object for training or testing.<br>
    Input: An optional boolean argument (default value is True for training dataset).<br>
    Returns: Dataloader for the training set (if training = True) or the test set (if training = False).<br>
    </li><br>

  <li>
    <b>build_model()</b>:A function to build the neural network with the following layers:<br>
    <ol>
      <li>A Flatten layer to convert the 2D pixel array to a 1D array.</li>
      <li>A Dense layer with 128 nodes and a ReLU activation.</li>
      <li>A Dense layer with 64 nodes and a ReLU activation.</li>
      <li>A Dense layer with 10 nodes.</li>
     </ol>
    Input: None.<br>
    Returns: An untrained neural network model.
 </li><br>
  
  <li>
    <b>train_model(model, train_loader, criterion, T)</b>: A function to train the neural network, print the model's accuracy per epoch, and print the model's accumulated loss (epoch loss/length of the dataset) per epoch.<br>
    Input: the model produced by the previous function, the train DataLoader produced by the first function, the criterion, and the number of epochs T for training.<br>
    Returns: None.
 </li><br>
 
  <li>
    <b>evaluate_model(model, test_loader, criterion, show_loss=True)</b>: A function that prints the model's average loss (if the show_loss argument is true), and the model's accuracy.<br>
    Input: the trained model produced by the previous function, the test DataLoader, and the criterion.<br>
    Returns: None.
 </li><br>
 
 <li>
    <b>predict_label(model, test_images, index)</b>: A function that prints the top 3 most likely labels for the image at the given index, along with their probabilities.<br>
    Input: The trained model and test images.<br>
    Returns: None.
 </li><br>
  
Additionally, I have also written the below function to allow the user to view the image from a particular index of the data_loader and to view its ground truth label. This is particularly useful for comparing the predicted label to the ground truth label, and for seeing the image that was being evaluated.
 
 <ul>
  <li>
    <b>visualize_image(data_loader, index)</b>: A function that saves the image corresponding to an index in the data_loader to an image file named 'image.png' and the ground truth label for that image.<br>
    Input: Dataloader for the test set and an image index.<br>
    Returns: None.<br>
    </li><br>
</ul>
  
  
</ul>
