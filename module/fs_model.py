from module.layer import *
from torch import nn
from module.sync_bn import SyncBatchNorm
from helper import context as ctx
from new_layer import FSLayer


class GNNBase(nn.Module):

    def __init__(self, layer_size, activation,fs:bool, use_pp=False, dropout=0.5, norm='layer', n_linear=0):
        super(GNNBase, self).__init__()
        self.fs = fs
        self.n_layers = len(layer_size) - 1
        self.layers = nn.ModuleList()
        self.activation = activation
        self.use_pp = use_pp
        self.n_linear = n_linear

        if norm is None:
            self.use_norm = False
        else:
            self.use_norm = True
            self.norm = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)

        # 统计值
        self.feature_zero = 0 # 特征中0的个数
        self.feature_fs_zero = 0 # 经过fs选择后，特征中0的个数
        self.feature_num = 0 # 特征的维度数


class GraphSAGEWithFS(GNNBase):

    def __init__(self, layer_size, activation, use_pp,
                 sigma, mu, fs: bool,
                 dropout=0.5, norm='layer', train_size=None, n_linear=0):
        super(GraphSAGEWithFS, self).__init__(layer_size, activation,fs, use_pp, dropout, norm, n_linear)
        """
            fs: whether to do feature selection or not
            layer_size[i]、layer_size[i+1]是第i层输入与输出的维度
        """
        # OPT1: 对于本地节点和边界节点的feature，原来的代码是每训练一次都通过update获取一次
        # 实际上，可以只在首次训练时获取，后续训练时不再获取
        self.inner_boundary_nodes_feat = None

        # OPT3: 本地存储边界节点的embedding，用于周期性更新
        # 如果有fs，0/1存储的都是None，否则都是tensor
        # 形状为[boundary_nodes_num, layer_size[i]], i>=2
        # 如果没有fs，0存储的是None
        self.local_stored_boundary_nodes_embedding = [None] * self.n_layers

        if self.fs:
            for i in range(self.n_layers):
                # fs层
                if i == 0:
                    self.layers.append(FSLayer(layer_size[0], sigma, mu))
                else:
                    # 非线性层
                    if i < self.n_layers - self.n_linear:
                        self.layers.append(GraphSAGELayer(layer_size[i], layer_size[i + 1], use_pp=use_pp))
                    # 线性层（一般没有）
                    else:
                        self.layers.append(nn.Linear(layer_size[i], layer_size[i + 1]))

                # norm
                if i < self.n_layers - 1 and self.use_norm:
                    if norm == 'layer':
                        self.norm.append(nn.LayerNorm(layer_size[i + 1], elementwise_affine=True))
                    elif norm == 'batch':
                        self.norm.append(SyncBatchNorm(layer_size[i + 1], train_size))
                use_pp = False

        else:
            for i in range(self.n_layers):
                # 非线性层
                if i < self.n_layers - self.n_linear:
                    self.layers.append(GraphSAGELayer(layer_size[i], layer_size[i + 1], use_pp=use_pp))
                # 线性层（一般没有）
                else:
                    self.layers.append(nn.Linear(layer_size[i], layer_size[i + 1]))

                # norm
                if i < self.n_layers - 1 and self.use_norm:
                    if norm == 'layer':
                        self.norm.append(nn.LayerNorm(layer_size[i + 1], elementwise_affine=True))
                    elif norm == 'batch':
                        self.norm.append(SyncBatchNorm(layer_size[i + 1], train_size))
                use_pp = False

    def forward(self, g, feat, in_deg=None, update_flag=False):
        # OPT3: update_flag只对train模式有效
        h = feat
        if self.fs:
            for i in range(self.n_layers):
                # fs层
                if i == 0:
                    # OPT1: 检查fs的输入
                    # 首次训练时获取本地节点和边界节点的feature
                    # 评估时，直接使用该节点的feature即可
                    if self.training:
                        # if self.inner_boundary_nodes_feat is None:
                        # 判断条件错误，inner_boundary_nodes_feat为None时确实需要更新
                        # 但实际上，应该是update_flag为True必须更新
                        if update_flag:
                            print("first training, get inner and boundary nodes feature")
                            self.inner_boundary_nodes_feat = ctx.buffer.update(i, h)

                        # OPT1: fs的输入是本地节点和边界节点的feature
                        h = self.inner_boundary_nodes_feat
                    
                    # 比较fs前后0数量的变化
                    self.feature_zero += zero_count(h)
                    before = h.shape
                    h = self.layers[0](h)
                    h = self.dropout(h)
                    after = h.shape
                    self.feature_fs_zero += zero_count(h)
                    self.feature_num += h.numel()
                    print("fs layer",before,after)
                    print(self.feature_num, self.feature_zero, self.feature_fs_zero)
                    
                    # # 比较fs前后0数量的变化
                    # self.feature_zero += zero_count(h)
                    # before = h.shape
                    # h = self.layers[0](h)
                    # h = self.dropout(h)
                    # after = h.shape
                    # self.feature_fs_zero += zero_count(h)
                    # self.feature_num += h.numel()
                    # print("fs layer",before,after)
                    # # fs layer torch.Size([9704, 500]) torch.Size([9704, 500])
                    # # fs layer torch.Size([10013, 500]) torch.Size([10013, 500])
                    # print(self.feature_num, self.feature_zero, self.feature_fs_zero)
                else:
                    # 非线性层
                    if i < self.n_layers - self.n_linear:
                        if self.training and (i > 0 or not self.use_pp):
                            # OPT2: fs的输出即第1层的输入不传
                            # 那么需要传第2到n-1层的输入embedding
                            if i == 1:
                                pass
                            else:
                                # OPT3: 并不是每轮训练都聚合embedding
                                # 不聚合embedding时，用本地存储的边界节点的embedding来更新
                                before = h.shape
                                before_h = h
                                # 使用其它进程存储的边界节点的embedding来更新
                                if update_flag:
                                    h = ctx.buffer.update(i, h)
                                    # 底层空间共用，但是不参与反向传播
                                    self.local_stored_boundary_nodes_embedding[i] = h[before_h.shape[0]:].detach()
                                    # print(h.requires_grad, self.local_stored_boundary_nodes_embedding[i].requires_grad)
                                    # True, False
                                # 使用本地存储的边界节点的embedding来更新
                                else:
                                    # concat涉及到h和self.local_stored_boundary_nodes_embedding[i]的计算
                                    # 因此需要将self.local_stored_boundary_nodes_embedding[i]也会参与反向传播
                                    # 但self.local_stored_boundary_nodes_embedding[i]可能在多轮训练中被使用，因此要把该变量从计算图中分离出来
                                    # 最简单的操作是直接分离结果h
                                    h = torch.concat([h, self.local_stored_boundary_nodes_embedding[i]], dim=0)
                                    h = h.detach().requires_grad_(True)

                                after = h.shape
                                # torch.unique(before_h[:921]==h[before_h.shape[0]:])
                                # tensor([False,  True], device='cuda:0')
                                # torch.unique(before_h==h[:before_h.shape[0]])
                                # tensor([True], device='cuda:0')
                                print("update",before,after)
                                # update torch.Size([9704, 500]) torch.Size([10562, 500])
                                # update torch.Size([10013, 500]) torch.Size([10934, 500])
                                # 9704是inner nodes，10562是inner+boundary nodes
                                
                                # 统计信息传输量
                        before = h.shape
                        h = self.dropout(h)
                        h = self.layers[i](g, h, in_deg)
                        after = h.shape
                        # non-linear layer torch.Size([10934, 500]) torch.Size([10013, 64])
                        # non-linear layer torch.Size([10562, 500]) torch.Size([9704, 64])
                        print("non-linear layer",before,after)
                        
                    # 线性层
                    else:
                        h = self.dropout(h)
                        h = self.layers[i](h)

                    # 对各层的输出归一化
                    # fs不需要，但self.norm[0]占了位
                    # logits不需要，也不产生归一化层
                    if i < self.n_layers - 1:
                        if self.use_norm:
                            h = self.norm[i](h)
                        h = self.activation(h)
        else:
            for i in range(self.n_layers):
                # 非线性层
                if i < self.n_layers - self.n_linear:
                    if self.training and (i > 0 or not self.use_pp):
                        # OPT1: 检查第0层的输入
                        # 首次训练时获取本地节点和边界节点的feature
                        # 评估时，直接使用该节点的feature即可
                        if i == 0:
                            # if self.inner_boundary_nodes_feat is None:
                            # 判断条件错误，inner_boundary_nodes_feat为None时确实需要更新
                            # 但实际上，应该是update_flag为True必须更新
                            if update_flag:
                                print("first training, get inner and boundary nodes feature")
                                self.inner_boundary_nodes_feat = ctx.buffer.update(i, h)

                            h = self.inner_boundary_nodes_feat
                            
                        # OPT2: feature即第0层的输入不传
                        # 那么需要传第1到n-1层的输入embedding
                        else:
                            # OPT3: 并不是每轮训练都聚合embedding
                            # 不聚合embedding时，用本地存储的边界节点的embedding来更新
                            before = h.shape
                            before_h = h
                            # 使用其它进程存储的边界节点的embedding来更新
                            if update_flag:
                                h = ctx.buffer.update(i, h)
                                self.local_stored_boundary_nodes_embedding[i] = h[before_h.shape[0]:].detach()
                            # 使用本地存储的边界节点的embedding来更新
                            else:
                                h = torch.concat([h, self.local_stored_boundary_nodes_embedding[i]], dim=0)                                 
                                h = h.detach().requires_grad_(True)

                            after = h.shape
                            # torch.unique(before_h[:921]==h[before_h.shape[0]:])
                            # tensor([False,  True], device='cuda:0')
                            # torch.unique(before_h==h[:before_h.shape[0]])
                            # tensor([True], device='cuda:0')
                            print("update",before,after)
                            # update torch.Size([9704, 500]) torch.Size([10562, 500])
                            # update torch.Size([10013, 500]) torch.Size([10934, 500])
                            # 9704是inner nodes，10562是inner+boundary nodes

                            # 统计信息传输量
                    before = h.shape
                    h = self.dropout(h)
                    h = self.layers[i](g, h, in_deg)
                    after = h.shape
                    # non-linear layer torch.Size([10934, 500]) torch.Size([10013, 64])
                    # non-linear layer torch.Size([10562, 500]) torch.Size([9704, 64])
                    print("non-linear layer",before,after)
                    
                # 线性层
                else:
                    h = self.dropout(h)
                    h = self.layers[i](h)

                # 对各层的输出归一化
                # fs不需要，但self.norm[0]占了位
                # logits不需要，也不产生归一化层
                if i < self.n_layers - 1:
                    if self.use_norm:
                        h = self.norm[i](h)
                    h = self.activation(h)

        return h


def zero_count(matrix: torch.Tensor) -> int:
    # 对0进行计数
    return torch.sum(matrix == 0).item()