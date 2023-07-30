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

'''
def normalize(adj):
    """Normalization by D^{-1/2} (A+I) D^{-1/2}."""
    rowsum = adj.sum(dim=1) + 1e-20
    d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    d_inv_sqrt[d_inv_sqrt == float('inf')] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt).to(adj.dtype)
    adj = adj.mm(d_mat_inv_sqrt).t().mm(d_mat_inv_sqrt)
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
def sampler(A, previous_nodes:list, sample_num:int):
    """
    A:torch.Tesor, 所有待选邻居节点（一个节点的所有邻居节点是包括它自己本身的）的邻接矩阵,
    行列数一样,对角线上都是1,即自己和自己连接
    previous_nodes: 上一层的节点在矩阵A中的index,而不是global id, 要求以在A中的ID从小到大的顺序排列
    sample_num:每层节点采样的最大值
    """
    U = A[previous_nodes,:]
    after_nodes = []
    for U_row in U:
        indices = U_row.indices
        sampled_indices = np.random.choice(indices, sample_num, replace=True)
        after_nodes.append(sampled_indices)
    after_nodes = np.unique(np.concatenate(after_nodes))
    # previous_nodes一定在after_nodes中
    after_nodes = np.concatenate([previous_nodes, np.setdiff1d(after_nodes, previous_nodes)])
    after_nodes = np.sort(after_nodes)
    adj = A[after_nodes, :][:, after_nodes]
    adj = normalize(adj)
    adj = adj[previous_nodes, :]
    #adj = row_normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj) #将adj转换为sparse tensor类型,可以直接用于训练

    previous_index = np.where(np.isin(after_nodes, previous_nodes))[0]

    # 返回的adj用于前向传播中,after_nodes用于下一层采样,list,对应A中的ID,不是原ID
    # previous_index是previous_nodes在after_nodes中的索引,后面训练时要用
    return adj, after_nodes, previous_index
'''

def row_normalize(tensor):
    row_sum = tensor.sum(dim=1, keepdim=True)
    normalized_tensor = tensor / row_sum
    return normalized_tensor

# 层采样
def layer_wise_sampler(A, previous_nodes:list, sample_num:int):
    '''
    A:torch.Tesor, 所有待选邻居节点（一个节点的所有邻居节点是包括它自己本身的）的邻接矩阵,
    行列数一样,对角线上都是1,即自己和自己连接
    previous_nodes: 上一层的节点在矩阵A中的index,而不是global id, 要求以在A中的ID从小到大的顺序排列
    sample_num:每层节点采样的最大值
    '''
    s_num = min(A.shape[0], sample_num)
    sampled_nodes = torch.randperm(A.shape[0])[:s_num].sort().values
    adj = A[previous_nodes, :][:, sampled_nodes]
    adj = row_normalize(adj)

    # previous_index = torch.where(torch.isin(sampled_nodes, torch.tensor(previous_nodes)))[0]

    # 返回的adj用于前向传播中,tensor.
    # sampled_nodes用于下一层采样,list,对应A中的ID,不是global ID
    return adj, sampled_nodes.tolist()

if __name__ == "__main__":
    data = None
    dataset = MyDataset(data)   #data是worker i上的训练集节点id

    batch_size = 128
    #生成批处理数据batch
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)