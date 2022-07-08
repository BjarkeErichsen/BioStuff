
import torch
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import createDataset
from torch.utils.data import Dataset, DataLoader
"""
import sys
for path in sys.path:
    print(path)
#print(torch.cuda.is_available())
#extraModule.printer()

print(Data)
"""

trainCats, trainDogs, testCats, testDogs = createDataset.create_datasets()
batch_size = 20
epochs = 1

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, 3)
        self.conv2 = nn.Conv2d(12, 6, 3)

        #self.conv3 = nn.Conv2d(200, 400, 2)
        #self.conv4 = nn.Conv2d(400, 400, 2)

        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(23064, 1000)
        self.fc2 = nn.Linear(1000, 2)
        self.drop1 = nn.Dropout(p=0.1)
        self.drop2 = nn.Dropout(p=0.1)

    def forward(self, x, training=True):

        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))

        #x = F.relu(self.conv3(x))
        #x = self.pool(F.relu(self.conv4(x)))

        x = x.view(x.size(0), -1)
        if training:
            x = self.drop1(x)
        x = F.relu(self.fc1(x))
        if training:
            x = self.drop2(x)
        x = F.relu(self.fc2(x))

        x = F.softmax(x)
        return x

class theDataset(Dataset):
    def __init__(self):
        self.x = torch.cat((trainCats, trainDogs))
        self.y = [torch.tensor([1,0], dtype=torch.float32) if i<trainCats.shape[0] else torch.tensor([0,1], dtype=torch.float32) for i in range(self.x.shape[0])] #cats = [1,0], dogs = [0,1]
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return self.x.shape[0]



def train_nn(dataset_train, lr=1e-3, epochs=4, batch_size=20, device = "cpu"):
    criterion = nn.CrossEntropyLoss()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #training with either cpu or cuda
    # model = model.to(device=device) #to send the model for training on either cuda or cpu

    model = ConvNet()
    model = model.to(device=device)  # to send the model for training on either cuda or cpu
    optimizer = optim.RMSprop(model.parameters(), lr=lr)

    for epoch in range(epochs):  # epoch -> GPU training time for 1 epoch is 12 min

        train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

        for i, data in enumerate(train, 0):
            inputs, labels = data

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if i%100:
                print(outputs, labels)
            loss.backward()
            optimizer.step()

    return model
dataset = theDataset()
model = train_nn(dataset, epochs=epochs)



class theTestDataset(Dataset):
    def __init__(self):
        self.x = torch.cat((testCats, testDogs))
        self.y = [torch.tensor([0], dtype=torch.float32) if i<testCats.shape[0] else torch.tensor([1], dtype=torch.float32) for i in range(self.x.shape[0])] #cats = [1,0], dogs = [0,1]
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return self.x.shape[0]

def testing(dataset, model):

    test = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    with torch.no_grad():
        correct = 0
        count = 0
        for i, data in enumerate(test):
            inputs, labels = data

            outputs = model(inputs, training=False)

            predictions = torch.argmax(outputs, 1) #0 is cat, 1 is dog
            if predictions.item()-labels.item() == 0:
                correct += 1
            print(predictions.item(),labels.item())
            count+=1
    return correct, count
dataset = theTestDataset()
correct, count = testing(dataset, model)

print(correct, count)



