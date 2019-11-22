import torch
import torch.nn as nn
import torch.nn.functional as F


class ChamferLoss(nn.Module):
    def __init__(self, input_channels, reduction="mean"):
        # TODO: repsect reduction rule
        super(ChamferLoss, self).__init__()
        self.input_channels = input_channels

    def forward(self, x, y):
        batch_size = x.shape[0]
        num_points = x.shape[1]
        x = x[:, :, : self.input_channels]
        y = y[:, :, : self.input_channels]
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        indices = torch.arange(0, num_points)
        rx = xx[:, indices, indices].unsqueeze(1).expand(xx.size())
        ry = yy[:, indices, indices].unsqueeze(1).expand(yy.size())
        pdist = rx.transpose(2, 1) + ry - 2 * zz
        loss = torch.mean(pdist.min(1)[0]) + torch.mean(pdist.min(2)[0])
        return loss
