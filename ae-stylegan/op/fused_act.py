import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function


def fused_leaky_relu(input, bias=None, negative_slope=0.2, scale=2 ** 0.5):
    return F.leaky_relu(input, negative_slope=0.2) * scale