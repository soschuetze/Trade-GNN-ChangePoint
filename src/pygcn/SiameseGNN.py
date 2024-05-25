import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv, global_sort_pool
from torch_geometric.nn.aggr import SortAggregation
from torch.nn import Linear, LayerNorm, ReLU, Sigmoid
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

class GNN(torch.nn.Module):
    def __init__(self, num_features):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_features, 128)
        self.conv2 = GCNConv(128, 64)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index.to(torch.int64), data.edge_attr
        x = self.conv1(x, edge_index, edge_attr).relu()
        x = self.conv2(x, edge_index, edge_attr).relu()
        return x

class SiameseGNN(torch.nn.Module):
    def __init__(self, num_features):
        super(SiameseGNN, self).__init__()
        self.gnn = GNN(num_features)
        self.sort_aggr = SortAggregation(k=50)

        self.fc1 = Linear(9950, 128)  # Adjust input size according to pooling output
        self.norm1 = LayerNorm(128)
        self.relu1 = ReLU()

        self.fc2 = Linear(128, 64)
        self.norm2 = LayerNorm(64)
        self.relu2 = ReLU()

        self.fc3 = Linear(64, 1)

    def forward(self, data1, data2):
        out1 = self.gnn(data1)
        out2 = self.gnn(data2)

        out = torch.cdist(out1, out2, p=2) #Euclidean distance
        out = self.sort_aggr(out, data1.batch) #Sort-k pooling layer
        out = out.view(out.size(0), -1)  # Flatten the pooled output

        #Fully Connected Layer 1
        out = self.fc1(out)
        out = self.norm1(out)
        out = self.relu1(out)

        #Fully Connected Layer 2
        out = self.fc2(out)
        out = self.norm2(out)
        out = self.relu2(out)

        out = self.fc3(out)
        out = torch.sigmoid(out)
        return out