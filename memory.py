import torch
from torch import nn
import torch.nn.functional as F


class Memory(nn.Module):
    def __init__(self, M, N):
        super(Memory, self).__init__()

        self.N = N
        self.M = M
        self.clean_addressing()
        self.rw_addressing = []

    def get_weights(self):
        return self.rw_addressing

    def clean_addressing(self):
        self.rw_addressing = []


class ReadHead(Memory):

    def __init__(self, M, N):
        super(ReadHead, self).__init__(M, N)
        print("--- Initialize Memory: ReadHead")

    def forward(self, memory, weights=None, idx=None):
        # Genera parametri
        if weights is None:
            weights = torch.zeros(1, self.M).cuda()
            weights[0, idx] = 1

        self.rw_addressing.append(weights[0])
        # Read
        read = torch.mm(weights, memory)
        return read
