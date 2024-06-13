import torch.nn as nn
from dgl.nn.pytorch.conv import GraphConv
import torch
import torch.nn.functional as F

class GCN(nn.Module):

    def __init__(self, input_dim, hidden_dim=16, layers=3, dropout=0.1, **kwargs):
        super(GCN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.nlayers = layers
        self.input_dim = input_dim

        self.layer0 = GraphConv(input_dim, hidden_dim)
        for i in range(layers-1):
            self.add_module('layer{}'.format(i + 1), GraphConv(hidden_dim, hidden_dim))

    def forward(self, graph):

        x, edge_index = graph.x, graph.edge_index.to(torch.int64)

        return x