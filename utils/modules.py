import torch
import torch.nn as nn
import torch.nn.functional as F
from . import config as cfg


class MyConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1, pooling=2):
        super(MyConv2d, self).__init__()
        self._conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                                padding=(kernel_size[0]//2, kernel_size[1]//2))
        self._batch_norm = nn.BatchNorm2d(out_channels)
        self._relu = nn.ReLU()
        self._max_pool = nn.MaxPool2d(pooling)

    def forward(self, x):
        x = self._conv2d(x)
        x = self._batch_norm(x)
        x = self._relu(x)
        out = self._max_pool(x)
        return out


class MyDense(nn.Module):
    def __init__(self, in_features, out_features):
        super(MyDense, self).__init__()
        self._dense = nn.Linear(in_features, out_features)
        self._relu = nn.ReLU()

    def forward(self, x):
        return self._relu(self._dense(x))



def split_to_chunk_with_hop(song, hop_size):
    tensor_list = []
    start, end = 0, 0
    i = 0
    while end < song.shape[1]:
        start = i * hop_size
        end = start + cfg.CHUNK_LEN
        tensor_list.append(song[:, start:end])
        i += 1

    return torch.stack(tensor_list)
