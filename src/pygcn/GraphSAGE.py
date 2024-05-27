import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, LayerNorm
from torch_geometric.nn.aggr import SortAggregation
from torch_geometric.nn import SAGEConv, global_sort_pool

class GraphSAGE(torch.nn.Module):
    def __init__(self, num_features):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(num_features, 128)
        self.conv2 = SAGEConv(128, 64)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index.to(torch.int64)
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return x
    
class SiameseGNN_SAGE(torch.nn.Module):
    def __init__(self, num_features):
        super(SiameseGNN_SAGE, self).__init__()
        self.gnn = GraphSAGE(num_features)
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
