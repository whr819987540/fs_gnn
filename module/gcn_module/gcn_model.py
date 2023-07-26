from torch import nn
from scipy import sparse
from new_layer import FSLayer
from module.gcn_module.gcn_layer import GraphConvolution
import torch


def matrix_transfer_volume(matrix: torch.Tensor) -> int:
    # 需要放到cpu上才能转化为稀疏矩阵
    if matrix.device != torch.device("cpu"):
        matrix = matrix.cpu()

    # 将矩阵转为稀疏矩阵, 然后返回传输时占用空间的大小（字节）
    sparse_matrix = sparse.csr_matrix(matrix.data)
    # print(matrix)
    # print(sparse_matrix)
    # print(sparse_matrix.data,sparse_matrix.indices,sparse_matrix.indptr)
    # print(sparse_matrix.data.nbytes,sparse_matrix.indices.nbytes,sparse_matrix.indptr.nbytes)
    return sparse_matrix.data.nbytes+sparse_matrix.indices.nbytes+sparse_matrix.indptr.nbytes


def zero_count(matrix: torch.Tensor) -> int:
    # 对0进行计数
    return torch.sum(matrix == 0).item()

# matrix_transfer_volume(torch.randint(0,10,(3,3)))


class GCN(nn.Module):
    def __init__(self, layer_size, dropout, sigma, mu, fs: bool = True):
        """
            fs: whether to do feature selection or not
        """
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.fs = fs

        # 如果有fs, layer_size应该是各个层的输出维度
        # 如果没有fs， layer_size[i]、layer_size[i+1]是第i层输入与输出的维度
        if self.fs:
            self.layers.append(FSLayer(layer_size[0], sigma, mu))

        for i in range(len(layer_size)-1):
            self.layers.append(GraphConvolution(layer_size[i], layer_size[i+1]))

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # 传输量
        # 模型的参数（稀疏矩阵） 字节数
        self.transfer_volume = 0

        self.feature_zero = 0  # feature中的0
        self.feature_fs_zero = 0  # fs后feature中的0
        self.feature_num = 0  # feature中数字的个数

        self.normal_layer_output_zero_num = 0

    def forward(self, g, x):
        """
            将feature以及每层的输出作为传输对象(预测值除外)
            以稀疏矩阵的方式进行传输, 比较有无fs情况下传输量的变化
        """
        # 稀疏矩阵转化为tensor后可以直接参与torch.spmm运算，而无需转为稠密矩阵
        adj = g.adjacency_matrix()
        adj_tensor = torch.sparse_coo_tensor(
            indices=adj.indices(),
            values=adj.val,
            size=adj.shape
        )
        for i in range(len(self.layers)):
            # 统计传输量
            # feature以及最终的输出都不需要传输
            if i != 0:  # 不传第0层的输入（feature）
                if self.fs and i == 1:  # 如果有FS，不传第1层的输入
                    pass
                else:
                    with torch.no_grad():
                        self.transfer_volume += matrix_transfer_volume(x.data)

            if self.fs and i == 0:
                # 比较fs前后0数量的变化
                self.feature_zero += zero_count(x)
                x = self.layers[0](x)
                x = self.dropout(x)
                self.feature_fs_zero += zero_count(x)
                self.feature_num += x.numel()
                print(self.feature_num, self.feature_zero, self.feature_fs_zero)

            else:
                x = self.layers[i](x, adj_tensor)
                x = self.relu(x)
                x = self.dropout(x)
                if i != len(self.layers) - 1:
                    self.normal_layer_output_zero_num += zero_count(x)
        return x
