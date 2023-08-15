import torch

def gini_impurity(labels):
    unique_labels, counts = torch.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    gini = 1 - torch.sum(probabilities ** 2)
    return gini

# 计算连续型变量的gini分数
def continous_feature_importance_gini(X:torch.Tensor, y:torch.Tensor):
    '''
    在没有经过划分的原图的train上计算，X是特征矩阵，y是label矩阵
    '''
    n_samples = X.shape[0]
    n_features = X.shape[1]
    importance = torch.zeros(n_features)

    for i in range(n_features):
        # 获取第i个特征的取值
        feature_values = X[:, i]
        sorted_feature = torch.sort(feature_values).values
        threshold = sorted_feature[(n_samples-1) // 2]

        left_labels = y[feature_values <= threshold]
        right_labels = y[feature_values > threshold]

        left_gini = gini_impurity(left_labels)
        right_gini = gini_impurity(right_labels)

        weighted_gini = (len(left_labels) * left_gini + len(right_labels) * right_gini) / n_samples

        importance[i] = 1 / (weighted_gini+1e-20)

    importance = (importance - torch.mean(importance)) / torch.std(importance)

    return importance