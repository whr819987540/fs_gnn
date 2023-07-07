import torch

# 创建稀疏矩阵
indices = torch.tensor([[0, 0, 1, 2],
                        [0, 2, 1, 2]], dtype=torch.long)
values = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float)
shape = (3, 3)
mat1 = torch.sparse_coo_tensor(indices, values, shape)
print(mat1.to_dense())

# 创建稠密矩阵
mat2 = torch.tensor([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]], dtype=torch.float)

# 执行稀疏矩阵乘法
result = torch.spmm(mat1, mat2)

print(result)