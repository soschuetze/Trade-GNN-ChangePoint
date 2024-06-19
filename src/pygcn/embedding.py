import torch.nn as nn
from torch_geometric.nn import GraphConv, global_mean_pool
import torch
import torch.nn.functional as F

class GCN(nn.Module):

    def __init__(self, input_dim, hidden_dim=16, dropout=0.1, **kwargs):
        super(GCN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.input_dim = input_dim

        self.gnn1 = GraphConv(input_dim, hidden_dim)
        self.gnn2 = GraphConv(hidden_dim, hidden_dim)

        self.relu = nn.ReLU()

    def forward(self, graph):

        x, edge_index = torch.tensor(graph.x).float(), graph.edge_index.to(torch.int64)
        x = self.gnn1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.gnn2(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)

        return x
