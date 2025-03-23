import torch
import torch.nn as nn

class NeuralNetworkModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetworkModel,self).__init__()
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.lin2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.lin1(x)
        out = self.relu1(out)
        out = self.lin2(out)

        return out
        