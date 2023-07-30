import torch
from torch import nn
from fs_layer import FSLayer
from module.graphsage_layer import GraphSageConvolution
from module.graphsage_layer import GraphConvolution

class Graphsage_first(nn.Module):
    def __init__(self, nfeat, nhid, num_classes, layers, dropout, weights:torch.Tensor,
                 fs : bool = False, random : bool = True, pretrain : bool = True):
        '''
        :param layers: 不包括FS层的层数
        :param fs: 默认为False，即不加FS层
        :param random: 默认为True，即随机初始化FS层参数，否则，用Gini初始化
        :param pretrain: 默认为True，即使用第一种预训练的方式，否则，是用第二种全程online的方法
        :param weights: FS层参数的初始值，random为True时，传0就行了，
        因为weights不会被使用，random为False时，传continous_feature_importance_gini的输出
        '''
        super(Graphsage_first, self).__init__()
        self.layers = layers
        self.nhid = nhid
        self.fs = fs
        self.random = random
        self.pretrain = pretrain

        self.gcs = nn.ModuleList()
        if self.fs:
            self.gcs.append(FSLayer(nfeat, weights, random, pretrain))
        self.gcs.append(GraphSageConvolution(nfeat, nhid))
        for _ in range(layers - 1):
            self.gcs.append(GraphSageConvolution(2 * nhid, nhid))
        self.gc_out = nn.Linear(nhid, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adjs, previous_indices):
        '''
        :param x: tensor，第一层节点的feature，预训练模式下是原feature，offline模式下是经过选择后的feature
        :param adjs: torch.tensor，第一维的维数为层数，顺序是从第一层的adj到最后一层的adj
        :param previous_indices: list，第一维的维数为层数，顺序是从第一层的previous_index到最后一层的previous_index
        '''
        if self.fs:
            # 预训练模式
            x = self.gcs[0](x) # fs层
            x = self.dropout(x)
            for ell in range(len(self.gcs)-1):
                x = self.gcs[ell+1](x, adjs[ell], previous_indices[ell])
                x = self.relu(x)
                x = self.dropout(x)
        else:
            # offline模式
            for ell in range(len(self.gcs)):
                x = self.gcs[ell](x, adjs[ell], previous_indices[ell])
                x = self.relu(x)
                x = self.dropout(x)
        x = self.gc_out(x)
        return x

class GCN_first(nn.Module):
    def __init__(self, nfeat, nhid, num_classes, layers, dropout, weights:torch.Tensor,
                 fs : bool = False, random : bool = True, pretrain : bool = True):
        '''
        :param layers: 不包括FS层的层数
        :param fs: 默认为False，即不加FS层
        :param random: 默认为True，即随机初始化FS层参数，否则，用Gini初始化
        :param pretrain: 默认为True，即使用第一种预训练的方式，否则，是用第二种全程online的方法
        :param weights: FS层参数的初始值，random为True时，传0就行了，
        因为weights不会被使用，random为False时，传continous_feature_importance_gini的输出
        '''
        super(GCN_first, self).__init__()
        self.layers = layers
        self.nhid = nhid
        self.fs = fs
        self.random = random
        self.pretrain = pretrain

        self.gcs = nn.ModuleList()
        if self.fs:
            self.gcs.append(FSLayer(nfeat, weights, random, pretrain))
        self.gcs.append(GraphConvolution(nfeat, nhid))
        for _ in range(layers - 1):
            self.gcs.append(GraphConvolution(nhid, nhid))
        self.gc_out = nn.Linear(nhid, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adjs):
        '''
        :param x: tensor，第一层节点的feature，预训练模式下是原feature，offline模式下是经过选择后的feature
        :param adjs: torch.tensor，第一维的维数为层数，顺序是从第一层的adj到最后一层的adj
        '''
        if self.fs:
            # 预训练模式
            x = self.gcs[0](x) # fs层
            x = self.dropout(x)
            for ell in range(len(self.gcs)-1):
                x = self.gcs[ell+1](x, adjs[ell])
                x = self.relu(x)
                x = self.dropout(x)
        else:
            # offline模式
            for ell in range(len(self.gcs)):
                x = self.gcs[ell](x, adjs[ell])
                x = self.relu(x)
                x = self.dropout(x)
        x = self.gc_out(x)
        return x

# 根据预训练结果选择offline模式下的特征
def FeatureSeclectOut(fsratio, weights:torch.Tensor, X:torch.Tensor):
    '''

    :param fsratio: 超参数，选择特征的百分比
    :param weights: 预训练模型FS层的参数
    :param X: inner node的特征矩阵
    :return: inner node经过选择后的特征矩阵，特征维度减小，后面训练用这个特征
    '''
    k = int(weights.numel() * fsratio)
    _, indices = weights.sort(descending=True)
    selected_dims = indices[:k]

    return X[:, selected_dims]