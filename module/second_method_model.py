import torch
from torch import nn
from new_layer_1 import FSLayer
from helper.graphsage_layer import GraphSageConvolution

# 第二种全程online的方式，这种方式下是一定要加FS层的，在训练模型之前做节点采样
class GCN_second(nn.Module):
    def __init__(self, nfeat, nhid, num_classes, layers, dropout, weights:torch.Tensor,
                 random: bool = True):
        '''
        :param layers: 不包括FS层的层数
        :param random: 默认为True，即随机初始化FS层参数，否则，用Gini初始化
        :param weights: FS层参数的初始值，random为True时，传0就行了，
        因为weights不会被使用，random为False时，传continous_feature_importance_gini的输出
        '''
        super(GCN_second, self).__init__()
        self.layers = layers
        self.nhid = nhid
        self.random = random

        self.gcs = nn.ModuleList()
        self.gcs.append(FSLayer(nfeat, weights, random, False))
        self.gcs.append(GraphSageConvolution(nfeat, nhid, use_lynorm=False))
        for _ in range(layers - 1):
            self.gcs.append(GraphSageConvolution(2 * nhid, nhid, use_lynorm=False))
        self.gc_out = nn.Linear(nhid, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, X:torch.Tensor, adjs:torch.Tensor, sampled_nodes:list, previous_indices):
        '''

        :param X: inner node的feature矩阵
        :param adjs: torch.sparse_coo_tensor，第一维的维数为层数，顺序是从第一层的adj到最后一层的adj
        :param sampled_nodes: 第一层节点的ID
        :return:
        '''
        X = self.gcs[0](X) # fs层，这里的fs层并不会降维，只是把某些维度置为0，以此减少通信

        # worker之间的feature交换，根据sampled_nodes决定需要从其他worker上拿来的节点的feature
        # 得到X，行数为len(sampled_nodes)，列数为原feature维度，只是有些维度的feature被置为0

        for ell in range(len(self.gcs)):
            x = self.gcs[ell](X, adjs[ell], previous_indices[ell])
            x = self.relu(x)
            x = self.dropout(x)
        x = self.gc_out(x)
        return x