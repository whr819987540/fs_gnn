import torch
from torch import nn
import math

class GraphSageConvolution(nn.Module):
    def __init__(self, n_in, n_out, bias=True):
        super(GraphSageConvolution, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in, n_out, bias=bias)
        self.reset_parameters()

    def forward(self, x, adj, previous_index):
        #out_node_num = adj.size(0)  # previous_nodes
        x = self.linear(x)  # 行sampled_nodes，列2*nhid->行sampled_nodes，列nhid
        support = torch.spmm(adj, x)  # adj:previous_nodes*sampled_nodes, x:sampled_nodes*nhid
        x = torch.cat([x[previous_index], support], dim=1)  # previous_nodes*2nhid
        return x

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.linear.weight.size(1))
        self.linear.weight.data.uniform_(-stdv, stdv)
        if self.linear.bias is not None:
            self.linear.bias.data.uniform_(-stdv, stdv)