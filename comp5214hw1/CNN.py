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

save_path = 'CNN.pth'

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

class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = LeNet5().to(device)

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


optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

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
        torch.save(net.state_dict(), save_path)
        print("saved weight to", save_path)
    
    print("Best Acc :", best_acc)
