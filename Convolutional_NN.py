import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

PATH_TO_TRAIN_DATA = '~/Documents/Vanderbilt/MATH3670/MNIST_DATA/TRAIN'
PATH_TO_TEST_DATA = '~/Documents/Vanderbilt/MATH3670/MNIST_DATA/TEST'

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), ])
trainset = datasets.MNIST(PATH_TO_TRAIN_DATA, download=True, train=True, transform=transform)
valset = datasets.MNIST(PATH_TO_TEST_DATA, download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64,
                                        shuffle=True)  # Batch size is the number of images we want in one go


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(4 * 4 * 50, 500),
            nn.Linear(500, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 4 * 4 * 50)
        return self.classifier(x)


myCNN = CNN()
criterion = nn.NLLLoss()  # together with LogSoftmax acts as a BCELoss
optimizer = optim.Adam(myCNN.parameters(), lr=0.001, betas=(0.9, 0.999))

epochs = 15


def train():
    for epoch in range(epochs):
        total_running_loss = 0
        for images, labels in trainloader:
            optimizer.zero_grad()
            output = myCNN(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_running_loss += loss.item()
        print("Epoch %s -- Training loss: %s %%\n" % (epoch, total_running_loss / len(trainloader)))


def test():
    correct, total = 0, 0
    for images, labels in valloader:
        for i in range(len(labels)):
            with torch.no_grad():
                logps = myCNN(images[i])

            ps = torch.exp(logps)
            probability = list(ps.numpy()[0])
            pred_label = probability.index(max(probability))
            true_label = labels.numpy()[i]
            if true_label == pred_label:
                correct += 1
            total += 1

    print("Number of Images Tested = %s\n" % total)
    print("Prediction Accuracy = %s" % (correct / total))


train()
test()
