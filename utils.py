import torch
from torch.nn import functional as F
import os


def cosdistance(x, y, offset=1e-5):
    x = x / torch.sqrt(torch.sum(x ** 2)) + offset
    y = y / torch.sqrt(torch.sum(y ** 2)) + offset
    return torch.sum(x * y)


def L2distance(x, y):
    return torch.sqrt(torch.sum((x - y) ** 2))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


class loss_vc(torch.nn.Module):
    def __init__(self):
        super(loss_vc, self).__init__()
        kernel = torch.ones(1, 1, 4, 4)
        self.weight = torch.nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        batch_size = x.shape[0]

        x1 = x[:, 0, :, :]
        x2 = x[:, 1, :, :]
        x3 = x[:, 2, :, :]
        x1 = F.conv2d(x1.unsqueeze(1), self.weight, stride=1).reshape(-1)
        x2 = F.conv2d(x2.unsqueeze(1), self.weight, stride=1).reshape(-1)
        x3 = F.conv2d(x3.unsqueeze(1), self.weight, stride=1).reshape(-1)
        x1 = torch.var(x1, unbiased=True)
        x2 = torch.var(x2, unbiased=True)
        x3 = torch.var(x3, unbiased=True)
        x = x1 + x2 + x3
        # x = F.conv2d(x,self.weight,stride=14)
        x = x.reshape(-1)
        x = torch.mean(x)
        return x
