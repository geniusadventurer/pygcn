"""

train.py：训练流程。现在的问题是跑得太慢，以及准确率并不高。但至少已经跑得通了！
接下来的目标，一是提高运行速度，二是和原有的代码实现对比提高准确率。
现在想到的办法是，如果没有标准化adj矩阵呢？毕竟求逆还是需要一些运算量。

"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch.optim as optim

from models import GCN
from pathlib import Path


# 加载数据，将特征矩阵中的各数据转换为tensor，将邻接表转换为邻接矩阵后转为tensor，转移数据
def load_data(root, feature_name, adj_name):
    print("Loading dataset...")
    # 特征矩阵
    feature = np.genfromtxt(root / feature_name, dtype=str)  # 读取数据
    nodes = list(feature[:, 0].astype(int))  # 取出所有编号
    nodes_idx = {j: i for i, j in enumerate(nodes)}  # 建立节点编号和节点索引的映射
    all_features = feature[:, 1: -1].astype(float)  # 切片，取出所有特征
    n_features = all_features.shape[1]  # 计算特征数目
    classes = set(feature[:, -1])  # 切片，取出所有类别，取唯一值
    n_classes = len(classes)  # 计算类别数目
    classes_idx = {j: i for i, j in enumerate(classes)}  # 建立类别和类别索引的映射
    classes_to_label = np.vectorize(classes_idx.get)(feature[:, -1]).flatten()  # 输出每个node对应的类别索引
    # 邻接矩阵
    adj = np.genfromtxt(root / adj_name, dtype=int)  # 读取数据
    adj_idx = np.vectorize(nodes_idx.get)(adj)  # map只能用于一维ndarray，多维用vectorize
    adj_mat = sp.coo_matrix(
        (np.ones(len(adj_idx)), (adj_idx[:, 0].flatten(), adj_idx[:, 1].flatten())),
        shape=(len(nodes), len(nodes)),
        dtype=int
    ).A  # coo_matrix转为np.array才能被输入Tensor
    # 用index划分数据集，这是除了用DataLoader之外的另一种数据集划分方法，私以为其实更简单，也不用自己去实现什么魔术方法
    train_idx = range(int(len(nodes) * 0.8))
    test_idx = range(int(len(nodes) * 0.8), len(nodes))
    # 转为tensor并转移到cuda上
    all_features = torch.Tensor(all_features).cuda()
    adj_mat = torch.Tensor(adj_mat).cuda()
    classes_to_label = torch.LongTensor(classes_to_label).cuda()  # 输出的标签也要转为Tensor，并且是long类型

    return all_features, n_features, classes_to_label, n_classes, adj_mat, train_idx, test_idx


# 训练。无论是训练还是测试都需要将整个模型跑完，然后再将训练集和测试集对应的索引拿出来进行效果的计算。
def train(model, optimizer, epochs, adj, feature, label, train_idx):
    print("Training data:")
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred_train = model(adj, feature)[train_idx]
        label_train = label[train_idx]
        loss = F.nll_loss(pred_train, label_train)  # nll_loss不需要去做手动独热转换，直接输入output的向量和按index索引后的labels就好了
        acc = torch.sum((pred_train.argmax(dim=1).eq(label_train).int())) / feature.shape[0]
        loss.backward()
        optimizer.step()
        print("epoch: {:>d}".format(epoch),
              "loss: {:.4f}".format(loss.item()),
              "acc: {:.4f}".format(acc.item()))


def test(model, adj, feature, label, test_idx):
    model.eval()
    with torch.no_grad():
        pred_test = model(adj, feature)[test_idx]
        label_test = label[test_idx]
        loss = F.nll_loss(pred_test, label_test)
        # argmax：取最大值下标；eq：逐元素判断相等；int：类型转换。
        acc = torch.sum((pred_test.argmax(dim=1).eq(label_test).int())) / feature.shape[0]
    print("Testing data:")
    print("loss: {:.4f}".format(loss.item()),
          "acc: {:.4f}".format(acc.item()))


def main():
    # 参数
    n_hidden = 16
    dropout_rate = 0.5
    random_seed = 1
    learning_rate = 0.01
    weight_decay = 5e-4
    epochs = 20
    root = Path('../data/cora')
    feature_name = 'cora.content'
    adj_name = 'cora.cites'

    # 初始化随机数
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # 加载数据
    all_features, n_features, classes_to_label, n_classes, adj_mat, train_idx, test_idx = \
        load_data(root, feature_name, adj_name)

    # 初始化模型
    model = GCN(n_features, n_hidden, n_classes, dropout_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 转移模型
    model.cuda()

    # 训练
    train(model, optimizer, epochs, adj_mat, all_features, classes_to_label, train_idx)

    # 测试
    test(model, adj_mat, all_features, classes_to_label, test_idx)


if __name__ == '__main__':
    main()
