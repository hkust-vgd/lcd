import torch
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):
    def __init__(self, reduction="mean"):
        # TODO: repsect reduction rule
        super(MSELoss, self).__init__()

    def forward(self, x, y):
        loss = torch.pow(x - y, 2)
        return loss.mean()
