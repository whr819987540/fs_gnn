from sampler import normalize, sparse_mx_to_torch_sparse_tensor

def sample_full(adj, layer_num:int):
    '''
    adj: 类型：由sp.coo_matrix生成的稀疏矩阵。全图（没有进行划分）上的邻接矩阵。对角线上是1
    layer_num:模型层数
    '''
    lap = normalize(adj)
    adjs = [sparse_mx_to_torch_sparse_tensor(lap) for _ in range(layer_num)] # 转换成了sparse类型，可以直接用于训练

    # 这个adjs可以直接输入到模型中，对应forward函数中的adjs
    return adjs