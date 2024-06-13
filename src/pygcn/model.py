import torch
from torch.nn import Linear, LayerNorm, ReLU
from torch_geometric.nn import GCNConv, global_sort_pool
from torch_geometric.utils import degree
from torch_geometric.nn.aggr import SortAggregation
from src.utils.graphs import laplacian_embeddings, random_walk_embeddings, degree_matrix
from torch_geometric.utils import to_networkx
import networkx as nx
import numpy as np
from torch import nn, optim
import torch.nn.functional as F

class GNN(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(400, 128)  # Input feature size is 400
        self.conv2 = GCNConv(128, hidden_units)

    def forward(self, x, edge_index, batch):
        x = torch.tensor(x, dtype=torch.float32)
        if x.size(1) != 400:  # Ensure correct padding size
            pad_size = (400 - x.size(1), 0, 400 - x.size(0), 0)
            x = F.pad(x, pad_size, "constant", 0)
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return x
    
class MLP(nn.Module):
    """MLP with linear output"""
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout=0.1):

        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout = nn.Dropout(dropout)

        self.linears = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        self.linears.append(nn.Linear(input_dim, hidden_dim))
        for layer in range(num_layers - 2):
            self.linears.append(nn.Linear(hidden_dim, hidden_dim))
        self.linears.append(nn.Linear(hidden_dim, output_dim))

        for layer in range(num_layers - 1):
            self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))


    def forward(self, x):

        h = x
        for i in range(self.num_layers - 1):
            h = F.relu(self.batch_norms[i](self.linears[i](h)))
            h = self.dropout(h)
        return self.linears[-1](h)

class GraphSiamese(nn.Module):

    def __init__(self, embedding: nn.Module, top_k, nlinear=2, nhidden=16, dropout=0.1):

        super(GraphSiamese, self).__init__()
        self.embedding = embedding
        self.input_dim = embedding.input_dim

        self.similarity = nn.PairwiseDistance()
        self.descending = True

        self.pooling_layer = torch.topk
        self.top_k = top_k
        self.nlinear = nlinear
        self.mlp = MLP(nlinear, top_k, nhidden, 1, dropout=dropout)

    def forward(self, graph1, graph2):

        graph1_encoding = self.embedding(graph1)
        graph2_encoding = self.embedding(graph2)

        graph1_encoding = torch.tensor(graph1_encoding, dtype=torch.float32)
        graph2_encoding = torch.tensor(graph2_encoding, dtype=torch.float32)

        # otherwise compute similarity/distance between the node-level embeddings
        similarity = self.similarity(graph1_encoding, graph2_encoding)

        x, _ = self.pooling_layer(similarity, k=self.top_k)
        x = x.squeeze()
        
        if x.dim() < 2:
            x = torch.unsqueeze(x, 0)
        x = self.mlp(x)

        x = torch.sigmoid(x)
        return x
