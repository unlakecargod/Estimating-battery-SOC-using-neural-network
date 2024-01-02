import torch
import torch.nn as nn
import torch.nn.functional as F


class LNet(nn.Module):
    """全连接层，线性网络"""
    def __init__(self, input_size):
        super(LNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 15)
        self.fc2 = nn.Linear(15, 1)
        self.fc3 = nn.Linear(2, 1)

    def forward(self, inp):
        x1 = F.relu(self.fc1(inp[:, :, 0]))
        x1 = F.relu(self.fc2(x1))
        x2 = F.relu(self.fc1(inp[:, :, 1]))
        x2 = F.relu(self.fc2(x2))
        x3 = torch.cat([x1, x2], -1)
        x3 = self.fc3(x3)
        return torch.squeeze(x3)