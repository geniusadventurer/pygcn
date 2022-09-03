"""

layers.py：预定义图卷积层的实现

"""

import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    自定义一个层时，__init__输入的通常是层的某些参数。实例化时就要立即指定这些参数。调用时再输入forward要求的参数，一般为输入的数据。
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))  # 设定参数
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))  # 设定参数
        else:
            self.register_parameter('bias', None)  # 或注册参数
        self.reset_parameters()  # 初始化参数值

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))  # weight的第二维求根做分母，作为stdv
        self.weight.data.uniform_(-stdv, stdv)  # tensor.uniform: 从均匀分布中抽样数值进行填充
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)  # 矩阵乘法
        output = torch.spmm(adj, support)  # 稀疏矩阵乘法
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):  # 魔术方法，输出属性
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
