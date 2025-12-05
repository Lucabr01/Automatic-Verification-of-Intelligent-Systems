import torch
import torch.nn as nn
import torch.nn.functional as F

class HVACPolicy(nn.Module):
    def __init__(self, obs_dim=37, action_dim=3, action_low=None, action_high=None):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_out = nn.Linear(64, action_dim)

    def forward(self, x):
        h = torch.tanh(self.fc1(x))
        h = torch.tanh(self.fc2(h))
        o = self.fc_out(h)
        return torch.tanh(o) 
