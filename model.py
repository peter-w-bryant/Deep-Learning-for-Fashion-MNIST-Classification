import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Building a Deep-Learning Model for Predicting Labels of Hand Written Images, using the Fashion-MNIST dataset

def get_data_loader(training = True):
# Input preprocessing: Specifying Transform
    custom_transform= transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    if(training == True):
         # The TRAIN SET contains images and labels to train the neural network
        train_set = datasets.FashionMNIST('./data',train=True, download=True,transform=custom_transform)
        loader = torch.utils.data.DataLoader(train_set, batch_size=64)

    else:
        # The TEST SET contains images and labels for model evaluation
        test_set = datasets.FashionMNIST('./data', train=False, transform=custom_transform)
        loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

    return loader


def build_model():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=784, out_features=128, bias=True),
        nn.ReLU(),
        nn.Linear(in_features=128, out_features=64, bias=True),
        nn.ReLU(),
        nn.Linear(in_features=64, out_features=10, bias=True)
    )
    return model

def train_model(model, train_loader, criterion, T):
    model.train()                                                # Put the Model in Training Mode
    loader_len = len(train_loader.dataset)                       # Compute the train DataLoader length
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # Define the Standard Gradient Decent Optimizer
    for epoch in range(T):                                       # loop over the dataset multiple times
        epoch_loss = 0.0
        predicted = 0
        total = 0
        correct = 0
        for i, (images, labels) in enumerate(train_loader, 0):   # loop over the mini-batches    
            opt.zero_grad()                                      # zero the parameter gradients
            y_predict = model(images)                            # Compute the Predictions
            loss = criterion(y_predict, labels)                  # Compute the Loss
            loss.backward()                                      # Compute the Gradients
            opt.step()                                           # Update the Weights
            _, predicted = torch.max(y_predict.data, dim = 1)    # Get the Predicted Class
            total += labels.size(0)                              # Get the Total Number of Labels
            correct += (predicted == labels).sum().item()        # Get the Total Number of Correct Predictions
            epoch_loss += loss.item() * 64                       # Compute Total Loss for the Current Epoch

        # Compute the Accumulated Loss (Epoch Loss / Length of Dataset) 
        print("Train Epoch: " + str(epoch), " Accuracy: " + str(correct) + "/" + str(total) + "(" + str("{:.2%}".format(correct/total)) + ")", " Loss: " + "{:.3f}".format(epoch_loss/loader_len))
        epoch_loss = 0.0

def evaluate_model(model, test_loader, criterion, show_loss = True):
    model.eval()                                                 # Put model in Evaluation Mode
    loader_len = len(test_loader.dataset)                        # Compute the train DataLoader length
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # Define the Standard Gradient Decent Optimizer
    with torch.no_grad():                                        # Loop over the test set, without computing gradients
        predicted = 0
        total = 0
        correct = 0
        total_loss = 0
        for i, (images, labels) in enumerate(test_loader, 0):                # Loop over the mini-batches
            y_predict = model(images)                                        # Compute the Predictions
            loss = criterion(y_predict, labels)                              # Compute the Loss
            _, predicted = torch.max(y_predict.data, dim = 1)                # Get the Predicted Class
            total += labels.size(0)                                          # Get the Total Number of Labels
            correct += (predicted == labels).sum().item()                    # Get the Total Number of Correct Predictions
            total_loss += loss.item() * 64                                   # Compute Total Loss for the Current Epoch
        if(show_loss == True):                                               # Print the Loss if show_loss = True
            print("Average loss: " + str(format(total_loss/total, ".4f")))   # Print the Average Loss, the Total loss/Total Number of Labels
        print("Accuracy: " + str("{:.2%}".format(correct/total)))            # Print the Accuracy, the Total Number of Correct Predictions/Total Number of Labels

def predict_label(model, test_images, index):
    class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']
    len_classes = len(class_names)
    images, labels = next(iter(test_images))
    test_image = images[index]
    y_predict = model(test_image)
    probs = F.softmax(y_predict, dim = 1)
    percentage_probs = np.zeros(len_classes)
    for i in range(len_classes):
        percentage_probs[i] = (probs[0][i]/1)
    largest_idxs = (-percentage_probs).argsort()[:3]
    for i in range(3):
        print(class_names[largest_idxs[i]] + ": " + str("{:.2%}".format(percentage_probs[largest_idxs[i]])))

def visualize_image(data_loader, index):
    class_names = ['T-shirt/top','Trouser','Pullover',      # Define the Class Names
                   'Dress','Coat','Sandal','Shirt',
                   'Sneaker','Bag','Ankle Boot'] 
    images, labels = next(iter(data_loader))                # Get the Images and Labels from the DataLoader
    image = images[index].squeeze()                         # Get the Image at the Specified Index
    label = labels[index]                                   # Get the Label at the Specified Index
    plt.imshow(image, cmap="gray")                          # Plot the Image
    plt.savefig("image.png")                                # Save the Image
    print("Ground Truth Label: " + class_names[label])      # Print the Ground Truth Label

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

