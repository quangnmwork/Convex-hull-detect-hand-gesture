import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import statistics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#inital
num_epochs=10
learning_rate = 0.001
batch_size = 64
classes = ('hello','like','ok')




transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224))])

directory = r'data\\train_data'
train_dataset = []
test_dataset = []

for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        # print(filename + " Labels:" +filename[-5])
        img = cv2.imread(os.path.join(directory, filename),0)
        #print(img)
        obj = (transform(img),int(filename[-5]))
        #print(transform(img))
        #print(transform(img).shape)
        train_dataset.append(obj)

directory = r'data\\test_data'
for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        # print(filename + " Labels:" +filename[-5])
        img = cv2.imread(os.path.join(directory, filename),0)
        obj = (transform(img),int(filename[-5]))
        test_dataset.append(obj)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                          shuffle=False)





class Classifier(nn.Module):
    def __init__(self):
        def __init__(self):
            super(Classifier, self).__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.layer2 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.fc = nn.Linear(48 * 48 * 32, 3)

        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = out.reshape(out.size(0), -1)
            out = self.fc(out)
            return out
#print(type(train_dataset[0]))

model=Classifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        print(outputs.shape)
        print(labels.shape)
        loss = criterion(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 768 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(3)]
    n_class_samples = [0 for i in range(3)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')
    for i in range(3):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')

