"""

layers.py：自己动手实现一个图卷积层。

"""

import torch
import torch.nn as nn
import math


# 自定义一个层时，init输入的通常是层的某些参数
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # weight和bias设定为参数
        self.weight = nn.Parameter(torch.zeros((in_features, out_features)))
        stdv = 1 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if bias:
            self.bias = nn.Parameter(torch.zeros((out_features,)))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.bias = nn.Parameter(torch.zeros((out_features,)))

    def forward(self, adj: torch.Tensor, feature: torch.Tensor, symmetry_normalize=False):
        identity = torch.eye(adj.shape[0]).cuda()
        D = torch.diag(torch.sum(adj, dim=0))  # 由邻接矩阵求度矩阵
        adj = adj + adj.T + identity
        if symmetry_normalize:  # 对称归一化
            D = torch.pow(D, -0.5)
            normalized_D = torch.where(torch.isinf(D), torch.zeros_like(D), D)
            adj = normalized_D @ adj @ normalized_D
        else:  # 非对称归一化
            D = torch.pow(D, -1)
            normalized_D = torch.where(torch.isinf(D), torch.zeros_like(D), D)
            adj = normalized_D @ adj
        out = adj @ feature @ self.weight + self.bias  # 卷积
        return out
