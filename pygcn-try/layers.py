"""

layers.py：自己动手实现一个图卷积层。

"""

import torch
import torch.nn as nn


# 自定义一个层时，init输入的通常是层的某些参数
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # weight和bias设定为参数
        self.weight = nn.Parameter(torch.rand((in_features, out_features)))
        if bias:
            self.bias = nn.Parameter(torch.rand((out_features,)))
        else:
            self.bias = nn.Parameter(torch.zeros((out_features,)))

    def forward(self, adj: torch.Tensor, feature: torch.Tensor):
        # 卷积操作。adj为邻接矩阵，feature为特征。输入进来的都应该是已经转换好的Tensor。利用度矩阵和邻接矩阵的关系求度矩阵。
        D = torch.diag(torch.sum(adj, dim=0))  # 由邻接矩阵求度矩阵
        (D_eigvals, D_eigvecs) = torch.linalg.eig(D)
        # 求度矩阵的D^(-1/2)。这里注意的是因为torch.linalg.eig出现了复数，因此要把它转化回浮点数，会丢掉虚部。这类数据的特性使得丢掉虚部是可以接受的。
        normalized_D = (D_eigvecs @ torch.diag(torch.sqrt(D_eigvals)) @ torch.linalg.inv(D_eigvecs)).float()
        idenetity = torch.eye(adj.shape[0]).cuda()
        adj = adj + adj.T + idenetity  # 将A对称化，再为A加上一个self-loop
        out = normalized_D @ adj @ normalized_D @ feature @ self.weight + self.bias  # 卷积
        return out
