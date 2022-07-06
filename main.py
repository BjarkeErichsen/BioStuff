
import sys
import torch
import extraModule
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os
"""
for path in sys.path:
    print(path)
#print(torch.cuda.is_available())
#extraModule.printer()

print(Data)
"""

directoryCatsTrain = r"C:\Users\EXG\OneDrive - Danmarks Tekniske Universitet\Skrivebord\BioProgrammering\Data\training_set\training_set\cats"
directoryDogsTrain = r"C:\Users\EXG\OneDrive - Danmarks Tekniske Universitet\Skrivebord\BioProgrammering\Data\training_set\training_set\dogs"
directoryCatsTest = r"C:\Users\EXG\OneDrive - Danmarks Tekniske Universitet\Skrivebord\BioProgrammering\Data\test_set\test_set\cats"
directoryDogsTest = r"C:\Users\EXG\OneDrive - Danmarks Tekniske Universitet\Skrivebord\BioProgrammering\Data\test_set\test_set\dogs"

trainDogsList, trainCatsList, testDogsList, testCatsList = [], [], [], []
size = 128, 128

for Directory in [directoryCatsTrain, directoryDogsTrain, directoryCatsTest, directoryDogsTest]:
    for filename in os.listdir(Directory):
        f = os.path.join(Directory, filename)
        # checking if it is a file
        if os.path.isfile(f) and f[-3:] == 'jpg':

            if "training_set\cats" in Directory:
                image = Image.open(f)  #if image.size != (512, 512):
                image = image.resize(size)

                transform = transforms.Compose([transforms.PILToTensor()])
                transform = transforms.PILToTensor()
                img_tensor = transform(image)
                trainCatsList.append(img_tensor)

            elif "training_set\dogs" in Directory:
                image = Image.open(f) #f = location of image
                image = image.resize(size) #size = tuple of desired size the image sh
                transform = transforms.Compose([transforms.PILToTensor()])
                transform = transforms.PILToTensor()
                img_tensor = transform(image)  #gets tensor
                trainDogsList.append(img_tensor)


            elif "test_set\cats" in Directory:
                image = Image.open(f)
                image = image.resize(size)
                transform = transforms.Compose([transforms.PILToTensor()])
                transform = transforms.PILToTensor()
                img_tensor = transform(image)
                testDogsList.append(img_tensor)

            elif "test_set\dogs" in Directory:
                image = Image.open(f)
                image = image.resize(size)
                transform = transforms.Compose([transforms.PILToTensor()])
                transform = transforms.PILToTensor()
                img_tensor = transform(image)
                testCatsList.append(img_tensor)

trainCats = torch.stack(trainCatsList)    #(image, channels, height, width)
trainDogs = torch.stack(trainDogsList)
testDogs = torch.stack(testDogsList)
testCats = torch.stack(testCatsList)



trainDogs.shape()