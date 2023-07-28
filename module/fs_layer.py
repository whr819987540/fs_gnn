from torch import nn
import torch

class FSLayer(nn.Module):
    def __init__(self, dim, weights, random=True, pretrain=True):
        #dim为这一层的维度，输入维度输出维度一样, weights是训练参数初始值，用continous_feature_importance_gini函数算,
        # random为True表示训练参数weights随机初始化，为False表示用continous_feature_importance_gini函数初始化

        super(FSLayer, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(dim))
        self.dim = dim
        self.pretrain = pretrain
        self.reset_parameters(weights,random)

    def reset_parameters(self, weights, random):   #初始化训练参数
        with torch.no_grad():
            if random:
                torch.nn.init.uniform_(self.weights, a=0, b=1)
            else:
                self.weights.data = weights

    def forward(self, x):
        s = torch.sigmoid(self.weights)
        if not self.pretrain:
            s = torch.round(s)
        return s * x
