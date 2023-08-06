import torch
from torch.utils.data import Dataset, DataLoader
from  typing import List

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample


# def normalize(adj):
#     """Normalization by D^{-1/2} (A+I) D^{-1/2}."""
#     rowsum = adj.sum(dim=1) + 1e-20
#     d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
#     d_inv_sqrt[d_inv_sqrt == float('inf')] = 0.
#     d_mat_inv_sqrt = torch.diag(d_inv_sqrt).to(adj.dtype)
#     adj = adj.mm(d_mat_inv_sqrt).t().mm(d_mat_inv_sqrt)
#     return adj


# def row_normalize(adj):
#     """Row-normalize sparse matrix"""
#     rowsum = np.array(adj.sum(1)).flatten()
#     d_inv = 1.0 / (np.maximum(1.0, rowsum))
#     d_mat_inv = sp.diags(d_inv, 0)
#     adj = d_mat_inv.dot(adj)
#     return adj

# def sparse_mx_to_torch_sparse_tensor(sparse_mx):
#     """Convert a scipy sparse matrix to a torch sparse tensor."""
#     sparse_mx = sparse_mx.tocoo().astype(np.float32)
#     if len(sparse_mx.row) == 0 and len(sparse_mx.col) == 0:
#         indices = torch.LongTensor([[], []])
#     else:
#         indices = torch.from_numpy(
#             np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
#     values = torch.from_numpy(sparse_mx.data)
#     shape = torch.Size(sparse_mx.shape)
#     return indices, values, shape

def row_normalize(tensor):
    row_sum = tensor.sum(dim=1, keepdim=True) + 1e-20
    normalized_tensor = tensor / row_sum
    return normalized_tensor

def normalize(adj):
    rowsum = adj.sum(dim=1) + 1e-20
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    adj = adj.to(d_mat_inv_sqrt.dtype)
    adj_normalized = adj.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
    return adj_normalized

# 节点采样
def node_wise_sampling(A:torch.Tensor, previous_nodes:List[int], sample_num:int):
    """
    A:torch.Tesor, 所有待选邻居节点（一个节点的所有邻居节点是包括它自己本身的）的邻接矩阵,
    行列数一样,对角线上都是1,即自己和自己连接
    previous_nodes: 上一层的节点在矩阵A中的index,而不是global id, 要求以在A中的ID从小到大的顺序排列
    sample_num:每个节点采样的节点数

    返回的adj用于前向传播中
    sampled_nodes用于下一层采样,对应A中的index,不是global id
    previous_index是previous_nodes在after_nodes中的索引,后面训练时要用,从第一层到最后一层组成一个list,传给Graphsage_first中的参数previous_indices
    """
    U = A[previous_nodes,:]
    sampled_nodes = []
    for U_row in U:
        indices = U_row.nonzero().flatten()
        sampled_indices = indices[torch.randperm(indices.shape[0])[:sample_num]]
        sampled_nodes.append(sampled_indices)
    sampled_nodes = torch.unique(torch.cat(sampled_nodes))
    sampled_nodes = torch.unique(torch.cat([torch.tensor(previous_nodes), sampled_nodes]), sorted=True)
    adj = U[:, sampled_nodes]
    adj = row_normalize(adj)

    previous_index = torch.where(torch.isin(sampled_nodes, torch.tensor(previous_nodes)))[0]


    return adj, sampled_nodes, previous_index

# 层采样
def layer_wise_sampling(A:torch.Tensor,previous_nodes:List[int],sample_num:int):
    '''
        A:torch.Tesor, 所有待选邻居节点（一个节点的所有邻居节点是包括它自己本身的）的邻接矩阵,
        行列数一样,对角线上都是1,即自己和自己连接
        previous_nodes: 上一层的节点在矩阵A中的index,而不是global id, 要求以在A中的ID从小到大的顺序排列
        sample_num:每层节点采样的最大值

        adj:adj用于前向传播中
        adj.dtype torch.float32
        sampled_nodes(torch.Tensor): 用于下一层采样,对应A中的index,不是global id
    '''
    s_num = min(A.shape[0], sample_num)
    sampled_nodes = torch.randperm(A.shape[0])[:s_num].sort().values
    adj = A[previous_nodes, :][:, sampled_nodes]
    adj = row_normalize(adj)

    # previous_index = torch.where(torch.isin(sampled_nodes, torch.tensor(previous_nodes)))[0]

    return adj, sampled_nodes

# 层重要性采样
def layer_importance_sampling(A:torch.Tensor, previous_nodes:List[int], sample_num:int):
    '''
    A:torch.Tesor, 所有待选邻居节点（一个节点的所有邻居节点是包括它自己本身的）的邻接矩阵,
    行列数一样,对角线上都是1,即自己和自己连接
    previous_nodes: 上一层的节点在矩阵A中的index,而不是global id, 要求以在A中的ID从小到大的顺序排列
    sample_num:每层节点采样的最大值

    adj:adj用于前向传播中
    adj.dtype torch.float32
    sampled_nodes(torch.Tensor): 用于下一层采样,对应A中的index,不是global id
    '''
    lap = normalize(A)
    lap_sq = torch.mul(lap, lap)
    pi = torch.sum(lap_sq[previous_nodes, :], dim=0)
    p = pi / torch.sum(pi)
    s_num = min(A.shape[0], sample_num)
    sampled_nodes = torch.multinomial(p, s_num, replacement=False)
    sampled_nodes = torch.sort(sampled_nodes)[0]
    adj = lap[previous_nodes, :][:, sampled_nodes]
    adj = torch.mul(adj, 1/p[sampled_nodes])
    adj = row_normalize(adj)

    return adj, sampled_nodes

if __name__ == "__main__":
    data = None
    dataset = MyDataset(data)   #data是worker i上的训练集节点id

    batch_size = 128
    #生成批处理数据batch
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)