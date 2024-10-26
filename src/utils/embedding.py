import torch.nn as nn
from torch_geometric.nn import GCNConv, GINConv, GATConv, SAGEConv
import torch
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, remove_self_loops
import numpy as np

class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP, BN and ReLU."""
    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = nn.BatchNorm1d(self.mlp.output_dim)

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = F.relu(h)
        return h

# MLP class remains the same
class MLP(nn.Module):
    """MLP with linear output"""
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout=0.1):
        """MLP layers construction
        Paramters
        ---------
        num_layers: int
            The number of linear layers
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        """
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.training = True
        self.dropout = dropout

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
                h = F.dropout(h, training=self.training, p = self.dropout)
            return self.linears[-1](h)

class GCN(nn.Module):

    def __init__(self, input_dim, type='gcn', hidden_dim=16, layers=3, dropout=0.1, identity=True, **kwargs):
        super(GCN, self).__init__()
        self.type = type
        self.nlayers = layers
        self.input_dim = input_dim
        self.identity = identity
        self.dropout = dropout

        if type == 'gcn':
            self.layer0 = GCNConv(input_dim, hidden_dim)
            for i in range(layers-1):
                self.add_module('layer{}'.format(i + 1), GCNConv(hidden_dim, hidden_dim))

        elif type == 'sage':
            self.layer0 = SAGEConv(input_dim, hidden_dim)
            for i in range(layers-1):
                self.add_module('layer{}'.format(i + 1), SAGEConv(hidden_dim, hidden_dim))

        elif type == 'gin':
            for layer in range(self.nlayers):
                if layer == 0:
                    mlp = MLP(2, input_dim, hidden_dim, hidden_dim, dropout)
                else:
                    mlp = MLP(2, hidden_dim, hidden_dim, hidden_dim, dropout)

                self.add_module('layer{}'.format(layer), GINConv(ApplyNodeFunc(mlp)))

        elif type == 'gat':
            self.layer0 = GATConv(input_dim, hidden_dim, heads=1, concat=False)
            for i in range(layers - 1):
                self.add_module('layer{}'.format(i + 1),
                                GATConv(hidden_dim, hidden_dim, heads=1, concat=False))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index.to(torch.int64)

        if self.identity:
            x = torch.from_numpy(np.eye(400*len(x))).float()
        else:
            x = torch.from_numpy(np.vstack(x)).float()

        for i in range(self.nlayers-1):
            x = torch.relu(self._modules['layer{}'.format(i)](x, edge_index))
            x = F.dropout(x, training=True, p = self.dropout)
        x = self._modules['layer{}'.format(self.nlayers-1)](x, edge_index)

        return x

