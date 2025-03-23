import torch
import torch.nn as nn

class Simple(nn.Module):
    def __init__(self):
        super(Simple, self).__init__()
        self.lin = nn.Linear(1,1)

    def forward(self, x):
        return self.lin(x)