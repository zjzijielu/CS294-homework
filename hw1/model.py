import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

class FFNet(nn.Module):

    def __init__(self, input_size, output_size, hidden_size=64):
        super(FFNet, self).__init__()
        # one hidden layer 
        self.fc1 = nn.Linear(input_size, 2*hidden_size)
        self.fc2 = nn.Linear(2*hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
