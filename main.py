
import sys
import torch
import extraModule
import numpy as np

for path in sys.path:
    print(path)
print(torch.cuda.is_available())

extraModule.printer()