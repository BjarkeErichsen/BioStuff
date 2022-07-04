
import sys
#sys.path.append(r"C:\Users\EXG\OneDrive - Danmarks Tekniske Universitet\Skrivebord\BioProgrammering\Data\training_set\training_set")
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




image = Image.open('iceland.jpg')


transform = transforms.Compose([
    transforms.PILToTensor()])
transform = transforms.PILToTensor()
img_tensor = transform(image)
