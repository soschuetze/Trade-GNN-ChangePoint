import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, LayerNorm
from torch_geometric.nn.aggr import SortAggregation
from torch_geometric.nn import GATConv, global_sort_pool
import torch.nn as nn

class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super(GAT, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=2)
        self.conv2 = GATConv(hidden_dim*2, hidden_dim, heads=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index = data.x.float(), data.edge_index.to(torch.int64)

        #x = torch.eye(400, 400)
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index).relu()
        return x

class SiameseGNN_GAT(torch.nn.Module):
    def __init__(self, top_k, input_dim, dropout, nhidden):
        super(SiameseGNN_GAT, self).__init__()
        self.gnn = GAT(input_dim, nhidden, dropout)
        self.topk_layer = torch.topk
        self.top_k = top_k
        self.dropout = nn.Dropout(dropout)
        self.similarity = nn.PairwiseDistance()

        self.fc1 = Linear(top_k, nhidden*2)  # Adjust input size according to pooling output
        self.norm1 = LayerNorm(nhidden*2)
        self.relu1 = ReLU()

        self.fc2 = Linear(nhidden*2, nhidden)
        self.norm2 = LayerNorm(nhidden)
        self.relu2 = ReLU()

        self.fc3 = Linear(nhidden, 1)

    def forward(self, data1, data2):
        out1 = self.gnn(data1)
        out2 = self.gnn(data2)

        similarity = self.similarity(out1, out2)
        out, _  = self.topk_layer(similarity, k=self.top_k)

        #Fully Connected Layer 1
        out = self.fc1(out)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.dropout(out)

        #Fully Connected Layer 2
        out = self.fc2(out)
        out = self.norm2(out)
        out = self.relu2(out)
        out = self.dropout(out)

        out = self.fc3(out)
        out = torch.sigmoid(out)
        return out