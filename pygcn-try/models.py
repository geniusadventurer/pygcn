"""

models.py：预定义图卷积神经网络的传播过程

"""

import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, n_features, n_hidden, n_class, dropout_rate):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(n_features, n_hidden)
        self.gc2 = GraphConvolution(n_hidden, n_class)
        self.dropout_rate = dropout_rate

    # 这里加载好的数据就必须是tensor了
    def forward(self, adj, feature):
        x = F.relu(self.gc1(adj, feature))  # 原代码将relu只放在第一次图卷积中
        x = F.dropout(x, self.dropout_rate, self.training)
        x = self.gc2(adj, x)
        x = F.log_softmax(x, dim=1)
        return x