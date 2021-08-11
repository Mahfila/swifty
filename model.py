import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self,feature_dim):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(feature_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu((self.fc2(x)))
        x = F.relu((self.fc3(x)))
        x = F.relu((self.fc4(x)))
        x = self.fc5(x)

        return x
