import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

PATH_TO_TRAIN_DATA = '~/Documents/Vanderbilt/MATH3670/MNIST_DATA/TRAIN'
PATH_TO_TEST_DATA = '~Documents/Vanderbilt/MATH3670/MNIST_DATA/TEST'

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), ])
trainset = datasets.MNIST(PATH_TO_TRAIN_DATA, download=True, train=True, transform=transform)
valset = datasets.MNIST(PATH_TO_TEST_DATA, download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64,
                                        shuffle=True)  # Batch size is the number of images we want in one go

dataiter = iter(trainloader)
images, labels = dataiter.next()
print(images.shape)  # [64, 1, 28, 28]
print(labels.shape)  # [64]
plt.imshow(images[0].numpy().squeeze(), cmap='gray_r')

INPUT_SIZE = 28 ** 2
OUTPUT_SIZE = 10  # 10 classes of digits


# implement a 2 hidden-layer vanilla neural network
class MNIST_Net(nn.Module):
    def __init__(self):
        super(MNIST_Net, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_features=INPUT_SIZE, out_features=128, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=OUTPUT_SIZE, bias=False),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, input):
        return self.main(input)


MyVanilla = MNIST_Net()
criterion = nn.NLLLoss()  # together with LogSoftmax acts as a BCELoss
optimizer = optim.Adam(MyVanilla.parameters(), lr=0.001, betas=(0.9, 0.999))

epochs = 15


def train():
    for epoch in range(epochs):
        total_running_loss = 0
        for images, labels in trainloader:
            images = images.view(images.shape[0], -1)
            optimizer.zero_grad()
            output = MyVanilla(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_running_loss += loss.item()
        print("Epoch %s -- Training loss: %s %%\n" % (epoch, total_running_loss/len(trainloader)))


def test():
    correct, total = 0, 0
    for images, labels in valloader:
        for i in range(len(labels)):
            img = images[i].view(1, 784)
            with torch.no_grad():
                logps = MyVanilla(img)

            ps = torch.exp(logps)
            probability = list(ps.numpy()[0])
            pred_label = probability.index(max(probability))
            true_label = labels.numpy()[i]
            if true_label == pred_label:
                correct += 1
            total += 1

    print("Number of Images Tested = %s\n" % total)
    print("Prediction Accuracy = %s" % (correct/total))

train()
test()