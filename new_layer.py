from scipy.stats import norm
import torch
from torch import nn
import numpy as np
import torch.distributions as dist


def loss_func(results, labels, lamda, sigma, model,fs:bool, args):
    if args is not None and args.dataset == 'yelp':
            loss_gcn = torch.nn.BCEWithLogitsLoss(reduction='sum')
    else:
        loss_gcn = torch.nn.CrossEntropyLoss(reduction='sum')

    if fs:
        loss_new = 0
        # 创建正态分布对象
        normal_dist = dist.Normal(0, 1)

        for mu_i in list(model.parameters())[0]:  # 获取新加层的参数
            loss_new += normal_dist.cdf(mu_i / sigma)

        loss_fs = lamda * loss_new
    else:
        loss_fs = 0
    
    return loss_gcn(results, labels) + loss_fs


class FSLayer(nn.Module):
    def __init__(self, dim, sigma, mu):
        # dim为这一层的维度，输入维度输出维度一样,sigma是可调参数
        # mu是训练参数初始值，用feature_importance_gini函数算
        super(FSLayer, self).__init__()
        self.mu = nn.Parameter(torch.Tensor(dim))
        self.sigma = sigma
        self.dim = dim
        self.reset_parameters(mu)

    def reset_parameters(self, mu):  # 初始化训练参数
        with torch.no_grad():
            self.mu.data = mu

    def forward(self, h):
        rou = torch.normal(0, self.sigma, size=(self.dim,),device=h.device)
        s = torch.clamp(self.mu + rou, 0, 1)
        return s * h

    # def forward(self, h):
    #     rou = np.random.normal(0, self.sigma, self.dim)
    #     s = np.clip(self.mu + rou, 0, 1)
    #     return s * h


# def gini_impurity(labels):
#     unique_labels, counts = np.unique(labels, return_counts=True)
#     probabilities = counts / len(labels)
#     gini = 1 - np.sum(probabilities ** 2)
#     return gini


def gini_impurity(labels):
    # 没有定义新的Tensor，所以不需要考虑labels的device
    unique_labels, counts = torch.unique(labels, return_counts=True)
    probabilities = counts.float() / len(labels)
    gini = 1 - torch.sum(probabilities ** 2)
    return gini



# def feature_importance_gini(X, y):
#     # 使用训练数据集生成
#     # X是节点特征矩阵，y是label，函数的结果是FSLayer中的mu，也就是训练参数的初始值
#     n_features = X.shape[1]
#     importance = np.zeros(n_features)
#     mu = np.zeros(n_features)

#     for i in range(n_features):
#         # 获取第i个特征的取值
#         feature_values = X[:, i]
#         unique_values = np.unique(feature_values)

#         # 计算每个取值的Gini impurity
#         impurities = []
#         for value in unique_values:
#             mask = feature_values == value
#             impurity = gini_impurity(y[mask])
#             weight = len(y[mask]) / len(y)
#             impurities.append(impurity * weight)

#         # 计算特征重要性
#         importance[i] = np.sum(impurities)
#         mu[i] = 1 / importance[i]

#     proportion = 1 / np.max(mu)
#     mu = mu * proportion

#     return torch.Tensor(mu)


def feature_importance_gini(X, y):
    # 定义了新的Tensor，importance、mu，但没有和X、y直接运算
    n_features = X.shape[1]
    importance = torch.zeros(n_features)
    mu = torch.zeros(n_features)

    for i in range(n_features):
        feature_values = X[:, i]
        unique_values = torch.unique(feature_values)

        impurities = []
        for value in unique_values:
            mask = feature_values == value
            impurity = gini_impurity(y[mask])
            weight = len(y[mask]) / len(y)
            impurities.append(impurity * weight)

        importance[i] = torch.sum(torch.tensor(impurities))
        mu[i] = 1 / importance[i]

    proportion = 1 / torch.max(mu)
    mu = mu * proportion

    return mu
