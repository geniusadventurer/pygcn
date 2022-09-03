"""

models.py：预定义图卷积神经网络的传播过程

"""

import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution


class GCN(nn.Module):
    # nfeat：features的数目；nhid：隐藏单元的数目；nclass：类别的数目；dropout：dropout率
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)  # 第一层图卷积
        self.gc2 = GraphConvolution(nhid, nclass)  # 第二层图卷积
        self.dropout = dropout  # dropout
        # GraphConvolution函数实际上是自己实现了torch.nn里并不自带的图卷积运算

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))  # 图卷积，ReLU
        x = F.dropout(x, self.dropout, training=self.training)  # dropout
        x = self.gc2(x, adj)  # 图卷积
        return F.log_softmax(x, dim=1)  # 对数softmax
