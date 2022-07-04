
import sys
import torch
import numpy as np

for path in sys.path:
    print(path)
print(torch.cuda.is_available())
