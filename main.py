
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
import wandb
from tqdm.notebook import tqdm
wandb.login()
torch.manual_seed(69)

config = dict(epochs=5, batch_size=20)

lr = 0.001

def model_pipeline(hyperparameters):

    #tell wandb to get started
    with wandb.init(project = "pytorch-demo", config = hyperparameters):
        config = wandb.config

        #make the model and optimization problem

        model, train_loader, test_loader, criterion, optimizer = make(config)

        train(model, train_loader, criterion, optimizer, config)

        correct, count = test(model, test_loader)
        model.train()  # turning back training mode
        print("correct ", correct, "count ",count)

    return model

def make(config):
    trainCats, trainDogs, testCats, testDogs = createDataset.create_datasets()
    traindataset = theDataset(trainCats, trainDogs)
    testdataset = theTestDataset(testCats, testDogs)

    train_loader = train = torch.utils.data.DataLoader(traindataset, batch_size=config["batch_size"], shuffle=True)
    test_loader =  test = torch.utils.data.DataLoader(testdataset, batch_size=1, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    model = ConvNet()
    model = model.to(device="cpu")  # to send the model for training on either cuda or cpu
    optimizer = optim.Adam(model.parameters(), lr=lr)

    return model, train_loader, test_loader, criterion, optimizer
"""
import sys
for path in sys.path:
    print(path)
#print(torch.cuda.is_available())
#extraModule.printer()

print(Data)
"""

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

    def forward(self, x, training=False):

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
        x = self.fc2(x)

        #x = F.softmax(x, dim=1)
        return x
class theDataset(Dataset):
    def __init__(self,trainCats, trainDogs):
        self.x = torch.cat((trainCats, trainDogs))
        #self.x = trainCats
        self.y = [torch.tensor([1,0], dtype=torch.float32) if i<trainCats.shape[0] else torch.tensor([0,1], dtype=torch.float32) for i in range(self.x.shape[0])] #cats = [1,0], dogs = [0,1]
        #self.y = [torch.tensor([1, 0], dtype=torch.float32) for i in range(self.x.shape[0])]
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return self.x.shape[0]
class theTestDataset(Dataset):
    def __init__(self, testCats, testDogs):
        self.x = torch.cat((testCats, testDogs))
        self.y = [torch.tensor([0], dtype=torch.float32) if i<testCats.shape[0] else torch.tensor([1], dtype=torch.float32) for i in range(self.x.shape[0])] #cats = 0, dogs = 1
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return self.x.shape[0]

def train(model, train_loader, criterion, optimizer, config, lr=1e-4):
    helper = 0

    wandb.watch(model, criterion, log="all", log_freq=10)

    for epoch in range(config["epochs"]):  #use tqdm
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            #if i==0:
                #print(int(torch.count_nonzero(torch.round(F.softmax(outputs, dim=1))==labels).item()/2))

            loss.backward()
            optimizer.step()

            helper +=1
            if helper%100 == 0:
                train_log(loss, epoch)

    return model
def test(model,test_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        count = 0
        for i, data in enumerate(test_loader):
            inputs, labels = data

            outputs = model(inputs, training=False)

            predictions = torch.argmax(F.softmax(outputs,dim=1)) #0 is cat, 1 is dog
            if predictions.item()-labels.item() == 0:
                correct += 1

            count+=1
        wandb.log({"correct": correct, "total":count})

        #maybe save the file using torch.onnx.export(model, images, "model.onnx) wandb.save(mode.onnx)
    return correct, count

def train_log(loss,epoch):

    loss = float(loss)
    wandb.log({"epoch": epoch, "loss":loss})

    print("loss is ", loss)



model_pipeline(config)

