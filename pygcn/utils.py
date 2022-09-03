"""

utils.py：用到的工具，包括独热编码、加载数据、构造对称矩阵、计算准确率、将稀疏矩阵转换为tensor

"""

import numpy as np
import scipy.sparse as sp
import torch


def encode_onehot(labels):  # 独热编码
    classes = set(labels)  # 把所有的标签取出唯一值
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}  # 效果是{class1: [1,0,0,0,0], class2: [0,1,0,0,0]}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)  # 转成np.array，利用map避免写for函数
    return labels_onehot  # 输出为np.array


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))  # 读取数据
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)  # 创建节点稀疏矩阵，第二列到倒数第二列为数据
    labels = encode_onehot(idx_features_labels[:, -1])  # 最后一列是标签

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)  # 提取index
    idx_map = {j: i for i, j in enumerate(idx)}  # 构建字典
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)  # 读取边
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)  # 转换为np.array
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # 构造节点间邻接矩阵：sp.coo_matrix((data, (row, col)), shape=(x, y))，将data按照对应位置的row和col给定的位置放进矩阵里，其他值为0

    # build symmetric adjacency matrix，无向图（若注释掉这一行则为有向图）
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # 原始矩阵 + 转置过来算出的矩阵 - 不转置算出的矩阵（去重）

    # 归一化
    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))  # adj在归一化之前，先引入自环

    # 数据集划分（划分的是index）
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    # 全部转为tensor
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    # 使用D^(-1)A构造对称矩阵
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))  # 获取稀疏矩阵非零元素的行列索引
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
