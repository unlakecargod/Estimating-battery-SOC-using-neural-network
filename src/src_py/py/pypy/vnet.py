import torch.nn as nn
import torch.nn.functional as F


class Vnet(nn.Module):
    """Vgg16网络"""
    def __init__(self, in_channels):
        super(Vnet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv1d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv1d(128, 128, 3, padding=1)
        self.conv5 = nn.Conv1d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv1d(256, 256, 3, padding=1)
        self.conv7 = nn.Conv1d(256, 512, 3, padding=1)
        self.conv8 = nn.Conv1d(512, 512, 3, padding=1)
        self.pool1 = nn.MaxPool1d(2, stride=2)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, inp):
        x = inp.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool1(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv6(x))
        x = self.pool1(x)
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv8(x))
        x = self.pool1(x)
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv8(x))
        # 此处注释掉一层池化层
        # x = self.pool1(x)
        x = x.view(-1, 512)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.output(x))
        return x
