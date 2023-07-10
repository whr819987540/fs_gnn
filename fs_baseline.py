import scipy.sparse as sp
import torch
from torch import nn

# 第一种


class init_feature:
    # 此类在训练之前对feature做选择，输入是原始feature，输出是经过feature selection后的feature mask
    # 每个client上单独选择feature，在后面每轮训练中feature mask不变
    def __init__(self, X, y, factor):
        # method是选feature的几种方法，X是原feature（训练集），factor是选择的feature比例
        self.X = X
        self.y = y
        self.factor = factor

    '''
    def random(self):  #随机选择feature
        size = self.X.shape[1]
        mask = np.zeros(size)
        indices = np.random.choice(size, int(size * self.factor), replace=False)
        mask[indices] = 1
        return mask
    '''

    '''
    def MI(self):  #用互信息选择feature
        mi = mutual_info_classif(self.X, self.y)
        indices = np.argsort(mi)[::-1]
        mask = np.zeros(len(mi))
        mask[indices[:len(mi) * self.factor]] = 1
        return mask
    '''

    def Gini(self):  # 用普通的Gini分数选择feature
        gini = feature_importance_gini(self.X, self.y)
        indices = np.argsort(gini)[::-1]
        mask = np.zeros(len(gini))
        mask[indices[:len(gini) * self.factor]] = 1
        return mask


# 第二种
# 用互信息选择feature的baseline不用上面的方法，用下面的神经网络训练一个特征掩码
# 不需要交换FM层的参数
class FeatureMask(nn.Module):
    def __init__(self, input_dim):
        super(FeatureMask, self).__init__()
        self.fc = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        mask = torch.sigmoid(self.fc(x))
        mask = torch.round(mask)
        return mask


def compute_mutual_information(node_features, labels):
    # 计算节点特征的分布
    node_feature_distribution = torch.softmax(node_features, dim=1)

    # 计算标签的分布
    label_distribution = torch.softmax(labels, dim=1)

    # 计算节点特征和标签的联合分布
    joint_distribution = torch.matmul(node_feature_distribution.t(), label_distribution)

    # 计算互信息
    mutual_information = torch.sum(joint_distribution * (torch.log2(joint_distribution) - torch.log2(
        node_feature_distribution.unsqueeze(2) * label_distribution.unsqueeze(1))))

    return mutual_information


# 训练FeatureMask网络的过程
feature_mask_model = FeatureMask(input_dim)
gnn_model = model()
feature_mask_optim = torch.optim.Adam(feature_mask_model.parameters())
gnn_model_optim = torch.optim.Adam(gnn_model.paramters())

for epoch in range(30):
    # feature selection
    feature_mask = feature_mask_model(X)

    # logits
    logits = gnn_model(X*feature_mask)

    # loss
    gnn_loss = gnn_loss_func(logits, y)

    # update gnn model
    pubmedgnn_model_optim.zero_grad()
    gnn_loss.backward()
    gnn_model_optim.step()

    # update feature mask model
    if epoch != 0 and epoch % 5 == 0:
        for inner_epoch in range(5):
            feature_mask = feature_mask_model(X)
            gnn_output = gnn_model(X, adj)  # 用最新的gnn模型得到预测标签
            mi = compute_mutual_information(X * feature_mask, gnn_output)
            feature_mask_loss = -mi
            feature_mask_optim.zero_grad()
            feature_mask_loss.backward()
            feature_mask_optim.step()


# 第三种
# 对节点采样，不用再做fs，所有特征都参与训练
def normalize(adj):
    rowsum = np.array(adj.sum(1)) + 1e-20
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt, 0)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj


class graphsaint_sampler:
    def __init__(self, adj, node_budget):
        # node_budget是选几个节点
        # 可以是按照比例来设置这个超参数
        self.adj = adj
        self.lap_matrix = normalize(adj + sp.eye(adj.shape[0]))

        lap_matrix_sq = self.lap_matrix.multiply(self.lap_matrix)
        p = np.array(np.sum(lap_matrix_sq, axis=0))[0]
        self.sample_prob = node_budget * p / p.sum()
        self.node_budget = node_budget

    def mini_batch(self, seed, X):
        np.random.seed(seed)
        sample_mask = np.random.uniform(
            0, 1, len(X)) <= self.sample_prob
        probs_nodes = self.sample_prob[sample_mask]
        sampled_nodes = X[sample_mask]
        adj = self.adj[sampled_nodes, :][:, sampled_nodes]

        return adj, sampled_nodes
