import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import sys

save_path = 'MLP.png'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Training Using", device)

criterion = nn.CrossEntropyLoss()

augmentation_transform = transforms.Compose(
    [   transforms.RandomPerspective(distortion_scale=0.1, p=0.5, interpolation=2),
        transforms.RandomAffine(degrees = (-10, 10), translate=(0.1,0.1), scale=(0.9,1.1), shear=(0.1,0.1)),
        transforms.ToTensor()])

transform = transforms.Compose(
    [   transforms.ToTensor()])

trainset = torchvision.datasets.MNIST(root = 'data/', train = True, transform = augmentation_transform, download = True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=8)

testset = torchvision.datasets.MNIST(root = 'data/', train = False, transform = transform, download = True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=8)

def get_network(number_neurons):

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(784, number_neurons)
            self.fc2 = nn.Linear(number_neurons, number_neurons)
            self.fc3 = nn.Linear(number_neurons, number_neurons)
            self.fc4 = nn.Linear(number_neurons, 10)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = self.fc4(x)
            return x

    return Net().to(device)

def train_and_val_fn(epoch, net, train, loader, criterion, optimizer):

    t = tqdm(loader, file=sys.stdout)
    if train:
        t.set_description('Epoch %i %s' % (epoch, "Training"))
        net.train()
    else:
        t.set_description('Epoch %i %s' % (epoch, "Validation"))
        net.eval()

    running_loss = 0.0
    total_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        inputs = torch.reshape(torch.squeeze(inputs), (-1,784))

        if train:
            optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        t.update()

    t.close()
    average_loss = float(total_loss/len(loader))
    acc = 100 * correct / total
    return average_loss, acc

acc = []
neurons = []

for i in range(2,9,1):

    number_neurons = 2 ** i
    print("-"*70)
    print("Testing Network with", str(number_neurons), "neurons")
    print("-"*70)
    net = get_network(number_neurons)
    optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, verbose=True)

    best_acc = 0.0

    for epoch in range(20):

        print("-"*50)

        average_train_loss, train_acc = train_and_val_fn(epoch = epoch, net = net, train = True, loader = trainloader, criterion=criterion, optimizer=optimizer)
        
        with torch.no_grad():
            average_val_loss, val_acc = train_and_val_fn(epoch = epoch, net = net, train = False, loader = testloader, criterion=criterion, optimizer=optimizer)

        print("Average Training Loss :", average_train_loss, "Training acc :", train_acc, "%")
        print("Average Test Loss :", average_val_loss,  "Validation acc :", val_acc, "%")

        scheduler.step()

        if best_acc < val_acc:
            best_acc = val_acc
        
        print("Best Acc :", best_acc)
    neurons.append(number_neurons)
    acc.append(best_acc)

plt.scatter(neurons,acc)
plt.xlabel('Neurons')
plt.ylabel('Accuracy')
plt.savefig(save_path, dpi=200,facecolor='white')

print("The result is saved to", save_path)