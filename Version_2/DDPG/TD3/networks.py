import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, _):
        super(Actor, self).__init__()

        self.bn_0 = nn.BatchNorm1d(state_dim)
        self.l1 = nn.Linear(state_dim, 512)
        self.l2 = nn.Linear(512, 456)
        self.bn_2 = nn.BatchNorm1d(456)
        self.l3 = nn.Linear(456, 128)
        self.l4 = nn.Linear(128, action_dim)

        self.l4_1 = nn.Linear(128, 64)
        self.bn_4 = nn.BatchNorm1d(64 + state_dim)
        self.l5_1 = nn.Linear(64+state_dim, 32)
        self.l6_1 = nn.Linear(32, action_dim)

    def forward(self, input):
        x = input
        x = self.bn_0(x)
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = self.bn_2(x)
        x = F.relu(x)
        x = self.l3(x)
        x = F.relu(x)
        return torch.tanh(self.l4(x))
        x = self.l4_1(x)
        x = F.relu(x)
        x = torch.cat((x, input), 1)
        x = self.bn_4(x)
        x = F.relu(self.l5_1(x))
        x = torch.tanh(self.l6_1(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.bn_0 = nn.BatchNorm1d(state_dim + action_dim) # TODO: normalize action on the input
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.bn_2 = nn.BatchNorm1d(300)
        self.l3 = nn.Linear(300, 300)
        self.l4 = nn.Linear(300, 50)
        self.bn_4 = nn.BatchNorm1d(50)
        self.l5 = nn.Linear(50, 1)

    def forward(self, x, u):
        x = torch.cat([x, u], 1)
        x = self.bn_0(x)
        x = F.relu(self.l1(x))
        x = self.l2(x)
        x = self.bn_2(x)
        x = F.relu(x)
        x = F.relu(self.l3(x))
        x = self.l4(x)
        x = self.bn_4(x)
        x = F.relu(x)
        x = self.l5(x)
        return x
