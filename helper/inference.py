from sampler import row_normalize
import torch

# 使用该函数生成全图的经过转换的邻接矩阵，得到的adjs用于前向传播得到所有节点的预测，
# 用测试集节点的预测和label计算准确率
# 这里之所以让全图进行前向传播得到所有节点的预测，是因为测试时为了得到稳定的预测结果选择不采样的方法，
# 这时候一层层找所有的邻居节点比较麻烦，而且最后得到的第一层节点很可能已经扩散到了全图，所以索性直接
# 用全图从第一层开始前向传播
def sample_full(adj:torch.Tensor, layer_num:int):
    '''
    adj: 全图（没有进行划分）上的邻接矩阵。对角线上是1
    layer_num:GNN模型层数，不包括FS层
    '''
    adj = row_normalize(adj)
    adjs = [adj for _ in range(layer_num)]

    # 这个adjs对应模型forward函数中的adjs
    return adjs