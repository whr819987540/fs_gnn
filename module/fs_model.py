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
        """
        for i in range(self.n_layers):
            # fs层
            if i==0 and self.fs:
                # 如果有fs, layer_size应该是各个层的输出维度
                # 如果没有fs， layer_size[i]、layer_size[i+1]是第i层输入与输出的维度
                self.layers.append(FSLayer(layer_size[0], sigma, mu))
                # OPT1: 对于本地节点和边界节点的feature，原来的代码是每训练一次都通过update获取一次
                # 实际上，可以只在首次训练时获取，后续训练时不再获取
                self.inner_boundary_nodes_feat = None
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

    def forward(self, g, feat, in_deg=None,update_flag=False):
        h = feat
        for i in range(self.n_layers):
            # fs层
            if i == 0 and self.fs:
                # OPT1: 检查fs的输入
                # 首次训练时获取本地节点和边界节点的feature
                # 评估时，直接使用该节点的feature即可
                if self.training:
                    if self.inner_boundary_nodes_feat is None:
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
                        # OPT2: fs的输出不传
                        if i == 1 and self.fs:
                            pass
                        else:
                            before = h.shape
                            h = ctx.buffer.update(i, h)
                            after = h.shape
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