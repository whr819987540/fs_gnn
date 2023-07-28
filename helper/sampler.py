import torch
import scipy.sparse as sp
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample

dataset = MyDataset(data)   #data是worker i上的训练集节点id

batch_size = 128
#生成批处理数据batch
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

def normalize(adj):
    """Normalization by D^{-1/2} (A+I) D^{-1/2}."""
    rowsum = np.array(adj.sum(1)) + 1e-20
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt, 0)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj

def row_normalize(adj):
    """Row-normalize sparse matrix"""
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = 1.0 / (np.maximum(1.0, rowsum))
    d_mat_inv = sp.diags(d_inv, 0)
    adj = d_mat_inv.dot(adj)
    return adj

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    if len(sparse_mx.row) == 0 and len(sparse_mx.col) == 0:
        indices = torch.LongTensor([[], []])
    else:
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return indices, values, shape

#采样函数
def sampler(seed:int, A, previous_nodes:list, sample_num:int):
    '''
    seed: 随机种子，为了使每一层采样产生的随机数不一样，每次用np.random.randint(2**10 - 1)产生一个随机数作为seed传入
    A:不是tensor，A是由sp.coo_matrix生成的稀疏矩阵，所有待选邻居节点（一个节点的所有邻居节点是包括它自己本身的）的邻接矩阵，
    行列数一样，对角线上都是1，即自己和自己连接
    previous_nodes: 上一层的节点在矩阵A中的ID，而不是原ID，要求以在A中的ID从小到大的顺序排列
    sample_num:每个节点采样的邻居节点数，超参数，这里设置为5
    '''
    np.random.seed(seed)
    U = A[previous_nodes,:]
    after_nodes = []
    for U_row in U:
        indices = U_row.indices
        sampled_indices = np.random.choice(indices, sample_num, replace=True)
        after_nodes.append(sampled_indices)
    after_nodes = np.unique(np.concatenate(after_nodes))
    after_nodes = np.concatenate([previous_nodes, np.setdiff1d(after_nodes, previous_nodes)])
    after_nodes = np.sort(after_nodes)
    adj = A[after_nodes, :][:, after_nodes]
    adj = normalize(adj)
    adj = adj[previous_nodes, :]
    #adj = row_normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj) #将adj转换为sparse tensor类型，可以直接用于训练

    previous_index = np.where(np.isin(after_nodes, previous_nodes))[0]

    # 返回的adj用于前向传播中，after_nodes用于下一层采样，list，对应A中的ID，不是原ID
    # previous_index是previous_nodes在after_nodes中的索引，后面训练时要用
    return adj, after_nodes, previous_index