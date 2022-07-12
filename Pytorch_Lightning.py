import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms


class ConvNet(pl.LightningModule):
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


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
