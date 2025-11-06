import torch
import torch.nn as nn
import torch.nn.functional as F


class LsNet(nn.Module):
    """LSTM网络"""
    def __init__(self, hidden_size, input_size, num_layers):
        super(LsNet, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(hidden_size = hidden_size,
                           input_size = input_size,
                           num_layers = num_layers,
                           batch_first = True)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 1)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        h0 = torch.zeros(self.num_layers, inputs.size(0), self.hidden_size).to(inputs.device)
        c0 = torch.zeros(self.num_layers, inputs.size(0), self.hidden_size).to(inputs.device)
        s_o, _ = self.rnn(inputs, (h0, c0))
        s_o = s_o[:, -1, :]
        #x = F.dropout(F.relu(self.fc1(s_o)))
        x = self.fc3(s_o)
        return torch.squeeze(x)