from helper.sampler import row_normalize, normalize
import torch


# 使用该函数生成全图的经过转换的邻接矩阵，得到的adjs用于前向传播得到所有节点的预测，
# 用测试集节点的预测和label计算准确率
# 这里之所以让全图进行前向传播得到所有节点的预测，是因为测试时为了得到稳定的预测结果选择不采样的方法，
# 这时候一层层找所有的邻居节点比较麻烦，而且最后得到的第一层节点很可能已经扩散到了全图，所以索性直接
# 用全图从第一层开始前向传播
def sample_full(adj: torch.Tensor, layer_num: int, sampling_method: str):
    """
    adj: 全图(没有进行划分)上的邻接矩阵。对角线上是1
    layer_num:GNN模型层数,不包括FS层
    sampling_method: 重要性采样(layer_importance_sampling),随机采样(layer_wise_sampling或node_wise_sampling)

    返回的adjs用于前向传播中
    previous_indices只有在node采样配套graphsage_first模型时使用,两种层采样配套GCN_first模型时不用管这个返回值
    """
    if sampling_method=="layer_importance_sampling":   # 因为是重要性采样,所以在测试时adj也要按照重要性的方式归一化,所以这里用normalize
        adj = normalize(adj)
    else:
        adj = row_normalize(adj)
    adjs = [adj for _ in range(layer_num)]

    previous_index = torch.arange(adj.shape[0]).tolist()
    previous_indices = [previous_index for _ in range(layer_num)]

    return adjs, previous_indices
