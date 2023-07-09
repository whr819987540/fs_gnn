import torch
import dgl

dataset = dgl.data.PubmedGraphDataset()
g = dataset[0]

sparse_adj_matrix = g.adjacency_matrix()
print(type(sparse_adj_matrix))

c = torch.sparse_coo_tensor(
    indices=sparse_adj_matrix.indices(),
    values=sparse_adj_matrix.val,
    size=sparse_adj_matrix.shape
)
