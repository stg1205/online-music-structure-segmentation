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


class PointerAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(PointerAttention, self).__init__()
        self._Wq = nn.Linear(hidden_dim, hidden_dim)
        self._Wk = nn.Linear(hidden_dim, hidden_dim)
        self._v = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, cluster_embeddings, onehot_mask):
        '''
        `cluster_embeddings`: (B, k+1, hidden_dim)
        `onehot_mask`: (B, k+1, num_clusters)
        '''
        e = torch.tanh(self._Wq(cluster_embeddings[:, -1, :]).unsqueeze(dim=1) + self._Wk(cluster_embeddings))  # B * k+1 * hidden_dim
        scores = self._v(e).transpose(-1, -2)  # out: B * 1 * k+1
        cluster_sum_scores = torch.matmul(scores, onehot_mask.float())  # out: (B, 1, num_clusters)
        cluster_num = torch.sum(onehot_mask, dim=1, keepdim=True)  # out: (B, 1, num_clusters)
        out = cluster_sum_scores / cluster_num
        out = torch.where(out.isnan(), torch.tensor(0, dtype=torch.float).to(out.device), out)
        #out = cluster_sum_scores
        out = out.squeeze(1)  # out: (B, num_clusters)
        # a = F.softmax(scores, dim=1)
        return out


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
