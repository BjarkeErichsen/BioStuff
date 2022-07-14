import pprint #endel af basic python
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




"""
metric = {
    'name': 'correct percentage',
    'goal': 'maximize'}
sweep_config['metric'] = metric   #not neccesarry unless doing baysian optimization
parameters_dict = {
    'optimizer': {
        'values': ['adam', 'sgd']
        },
    'dropout': {
          'values': [0.3, 0.4, 0.5]
        }
    }
"""

sweep_config = {
    'method': 'bayes'
    }
metric = {
    'name': 'Final correctness rate',
    'goal': 'maximize'}
sweep_config['metric'] = metric
parameters_dict = {
    'lr': {"distribution": 'uniform', 'min':0, 'max': 0.1},
    'batch_size': {"values": [16, 32, 64]},
    'epochs': {'value':4}
    }

sweep_config['parameters'] = parameters_dict
sweep_id = wandb.sweep(sweep_config, project = "CatsAndDogs3")


"""
early_terminate = {'early_terminate':    {'type': 'hyperband', 'min_iter': 20}}
sweep_config['early_terminate'] = early_terminate"""
"""
parameters_dict = {
    'lr': {'values': [0.0001, 0.001, 0.01]},
    'batch_size': {'values': [16, 32, 64]},
    'epochs': {'value':4}
    }"""
def model_pipeline(config = None):



    #tell wandb to get started
    with wandb.init(project = "CatsAndDogs3", config = config):

        config = wandb.config
        #make the model and optimization problem

        model, train_loader, test_loader, criterion, optimizer = make(config)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        train(model, train_loader, test_loader, criterion, optimizer, config, device)
        correct, count = test(model, test_loader, device)
        wandb.log({"Final correctness rate": correct / count})
        #correct, count = test(model, test_loader, device)
    return model

def make(config):
    trainCats, trainDogs, testCats, testDogs = createDataset.create_datasets()
    traindataset = theDataset(trainCats, trainDogs)
    testdataset = theTestDataset(testCats, testDogs)

    train_loader = train = torch.utils.data.DataLoader(traindataset, batch_size=config["batch_size"], shuffle=True)
    test_loader =  test = torch.utils.data.DataLoader(testdataset, batch_size=1, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    model = ConvNet()
    model = model  # to send the model for training on either cuda or cpu
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

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

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))

        #x = F.relu(self.conv3(x))
        #x = self.pool(F.relu(self.conv4(x)))

        x = x.view(x.size(0), -1)
        x = self.drop1(x)
        x = F.relu(self.fc1(x))
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

def train(model, train_loader,test_loader, criterion, optimizer, config, device):
    helper = 0

    wandb.watch(model, criterion, log="all", log_freq=1000)

    for epoch in range(config["epochs"]):  #use tqdm
        complete_loss = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            #if i==0:
                #print(int(torch.count_nonzero(torch.round(F.softmax(outputs, dim=1))==labels).item()/2))

            loss.backward()
            optimizer.step()
            complete_loss += loss

        loss = float(loss)
        print("loss is ", loss)
        correct, count = test(model,test_loader, device)
        wandb.log({"epoch": epoch, "loss": loss, "percentage correct": correct / count})
    return model
def test(model,test_loader, device):
    model.eval()
    with torch.no_grad():
        correct = 0
        count = 0
        for i, data in enumerate(test_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            predictions = torch.argmax(F.softmax(outputs,dim=1)) #0 is cat, 1 is dog
            if predictions.item()-labels.item() == 0:
                correct += 1

            count+=1


        #maybe save the file using torch.onnx.export(model, images, "model.onnx) wandb.save(mode.onnx)
    model.train()
    return correct, count






#model_pipeline(config)
print('cuda' if torch.cuda.is_available() else 'cpu')
wandb.agent(sweep_id, model_pipeline, count=15) #count = number of random combinations of hyperparams



