

def printer():
    print("printing")

import torch

c = torch.tensor([4, 5, 6])

a = [c,c,c]

ab = torch.stack(a, 0)

size = 128, 128
