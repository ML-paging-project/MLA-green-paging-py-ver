import torch
import torch.nn as nn

import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, window_size, num_of_box_kind):
        super().__init__()
        self.conv1 = nn.Conv1d(window_size + 4, 512, 1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(256, 128, 1)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, num_of_box_kind)
        # print('.........'+str(num_of_box_kind))

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(torch.transpose(x, 1, 2))
        x = self.conv2(torch.transpose(x, 1, 2))
        x = self.pool(torch.transpose(x, 1, 2))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

v = [i for i in range(104)]
t = torch.FloatTensor([[v]]).to(device)
t = torch.transpose(t, 1, 2)
net = DQN(100, 4).to(device)
print(net(t))
