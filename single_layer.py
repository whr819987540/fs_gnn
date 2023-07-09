import torch
import torch.nn as nn
import math


class GraphConvolution(nn.Module):
    def __init__(self, n_in, n_out, bias=True):
        super(GraphConvolution, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in, n_out)
        self.reset_parameters()

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        # torch.spmm(sparse/dense, dense)
        x = self.linear(x)
        x = torch.spmm(adj, x)
        return x

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.linear.weight.size(1))
        self.linear.weight.data.uniform_(-stdv, stdv)
        if self.linear.bias is not None:
            self.linear.bias.data.uniform_(-stdv, stdv)
