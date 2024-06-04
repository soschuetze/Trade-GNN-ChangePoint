import torch
from torch.nn import Linear, LayerNorm, ReLU
from torch_geometric.nn import GCNConv, global_sort_pool
from torch_geometric.utils import degree
from torch_geometric.nn.aggr import SortAggregation

class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(1, 128)  # Input feature size is 1 (node degree)
        self.conv2 = GCNConv(128, 64)

    def forward(self, data):
        # Use node degrees as features
        edge_index = data.edge_index.to(torch.int64)
        x = degree(data.edge_index[0], dtype=torch.float).view(-1, 1)
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return x

class SiameseGNN(torch.nn.Module):
    def __init__(self):
        super(SiameseGNN, self).__init__()
        self.gnn = GNN()
        self.sort_aggr = SortAggregation(k=50)

        self.fc1 = Linear(20000, 128)  # Adjust input size according to pooling output
        self.norm1 = LayerNorm(128)
        self.relu1 = ReLU()

        self.fc2 = Linear(128, 64)
        self.norm2 = LayerNorm(64)
        self.relu2 = ReLU()

        self.fc3 = Linear(64, 1)

    def forward(self, data1, data2):
        out1 = self.gnn(data1)
        out2 = self.gnn(data2)

        out = torch.cdist(out1, out2, p=2)  # Euclidean distance
        out = self.sort_aggr(out, data1.batch) #Sort-k pooling layer
        out = out.view(out.size(0), -1)  # Flatten the pooled output

        # Fully Connected Layer 1
        out = self.fc1(out)
        out = self.norm1(out)
        out = self.relu1(out)

        # Fully Connected Layer 2
        out = self.fc2(out)
        out = self.norm2(out)
        out = self.relu2(out)

        out = self.fc3(out)
        out = torch.sigmoid(out)
        return out