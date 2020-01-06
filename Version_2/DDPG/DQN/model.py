import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        # self.seed = torch.manual_seed(seed)
        # self.fc1 = nn.Linear(state_size, fc1_units)
        # self.fc2 = nn.Linear(fc1_units, fc2_units)
        # self.fc3 = nn.Linear(fc2_units, action_size)

        self.bn_0 = nn.BatchNorm1d(state_size)
        self.l1 = nn.Linear(state_size, 512)
        self.l2 = nn.Linear(512, 456)
        self.bn_2 = nn.BatchNorm1d(456)
        self.l3 = nn.Linear(456, 128)
        self.l4 = nn.Linear(128, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        # x = F.relu(self.fc1(state))
        # x = F.relu(self.fc2(x))
        # return self.fc3(x)
        x = state
        x = self.bn_0(x)
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = self.bn_2(x)
        x = F.relu(x)
        x = self.l3(x)
        x = F.relu(x)
        return self.l4(x)



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