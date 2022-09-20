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
    <b>evaluate_model(model, test_loader, criterion, show_loss=True)</b>: A function that prints the model's average loss (if the show_loss argument is true), and the model's accuracy on the testing data set.<br>
    Input: the trained model produced by the previous function, the test DataLoader, and the criterion.<br>
    Returns: None.
 </li><br>
 
 <li>
    <b>predict_label(model, test_images, index)</b>: A function that prints the top 3 most likely labels for the image at the given index, along with their probabilities.<br>
    Input: The trained model and test images.<br>
    Returns: None.
 </li>
</ul>
  
Additionally, I have also written the below function to allow the user to view the image from a particular index of the data_loader and to view its ground truth label. This is particularly useful for comparing the predicted label to the ground truth label, and for seeing the image that was being evaluated.
 
<ul>
  <li>
    <b>visualize_image(data_loader, index)</b>: A function that saves the image corresponding to an index in the data_loader to an image file named 'image.png' and prints the ground truth label for that image.<br>
    Input: Dataloader for the test set and an image index.<br>
    Returns: None.<br>
    </li><br>
</ul>
  
## Sample Output

Below are a few examples of using the functions above for different images trained for varying numbers of epochs. 

### 1.1 Main Method
```python
if __name__ == '__main__':
    # 1. get_data_loader()
    train_loader = get_data_loader()
    test_loader = get_data_loader(training = False)

    # 2. build_model()
    model = build_model()

    # 3. train_model()
    criterion = nn.CrossEntropyLoss()
    train_model(model, train_loader, criterion, T = 5)

    # 4. evaluate_model()
    evaluate_model(model, test_loader, criterion, show_loss = True)

    # 5. predict_label()
    predict_label(model, test_loader, 1)

    # Testing - visualize_image()
    visualize_image(test_loader, 1)
    pass
```
### 1.2 Linux Output
![sample1](https://user-images.githubusercontent.com/72423203/191143638-d30aac9e-e010-4056-a4b9-b6e7cb682bee.png)

In this instance, 

<i>
Train Epoch: 0  Accuracy: 41070/60000(68.45%)  Loss: 0.915<br>
Train Epoch: 1  Accuracy: 49319/60000(82.20%)  Loss: 0.513<br>
Train Epoch: 2  Accuracy: 50369/60000(83.95%)  Loss: 0.455<br>
Train Epoch: 3  Accuracy: 51044/60000(85.07%)  Loss: 0.423<br>
Train Epoch: 4  Accuracy: 51468/60000(85.78%)  Loss: 0.401<br>
</i>

corresponds to the output of the <b>train_model()</b> function. Showing us that on our 5th epoch my model achieves 85.78% prediction accuracy on the training dataset. 

<i>
Average loss: 0.4301<br>
Accuracy: 84.48%<br>
</i>

corresponds to the output of the <b>evaluate_model</b> function. Showing us that after training for 5 epochs, my model achieves 84.48% prediction accuracy on the testing dataset with an average loss of 0.4301.

<i>
Pullover: 92.45%<br>
Shirt: 6.20%<br>
Coat: 1.23%
</i><br><br>

corresponds to the output of the <b>predict_label</b> function. Showing us the model's top three predictions for the class that the image belongs to, along with their probability.

<i>
Ground Truth Label: Pullover
</i><br><br>

corresponds to the output of the <b>visualize_image</b> function. Showing us the ground truth label for the particular image our model just evaluated, and <br>

![image](https://user-images.githubusercontent.com/72423203/191144771-0b89bc19-f125-41a6-bba2-1d8b06c63977.png)<br>

is the image that was saved to 'image.png' showing us the original image that the model was evaluating.
