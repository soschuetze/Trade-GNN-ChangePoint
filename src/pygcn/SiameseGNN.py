import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv, SAGEConv, global_sort_pool
from torch_geometric.nn.aggr import SortAggregation
from torch.nn import Linear, LayerNorm, ReLU, Sigmoid, BatchNorm1d
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super(GNN, self).__init__()
        self.conv1 = SAGEConv(input_dim, 128)
        self.conv2 = SAGEConv(128, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index).relu()
        return x

class SiameseGNN(torch.nn.Module):
    def __init__(self, top_k, input_dim, dropout, nhidden):
        super(SiameseGNN, self).__init__()
        self.gnn = GNN(input_dim, nhidden, dropout)
        self.topk_layer = torch.topk
        self.top_k = top_k
        self.dropout = nn.Dropout(dropout)
        self.similarity = nn.PairwiseDistance()

        self.fc1 = Linear(top_k, 128)  # Adjust input size according to pooling output
        self.norm1 = BatchNorm1d(128)
        self.relu1 = ReLU()

        self.fc2 = Linear(128, nhidden)
        self.norm2 = BatchNorm1d(nhidden)
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