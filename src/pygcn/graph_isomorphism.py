import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, LayerNorm, Sequential
import torch.nn as nn
from torch_geometric.nn import GINConv, global_sort_pool, SortAggregation, global_mean_pool

class GIN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim):
        super(GIN, self).__init__()
        
        # Define MLP for GINConv layers
        self.mlp1 = torch.nn.Sequential(
            torch.nn.Linear(num_features, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.mlp2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.mlp3 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Define GINConv layers
        self.conv1 = GINConv(self.mlp1)
        self.conv2 = GINConv(self.mlp2)
        self.conv3 = GINConv(self.mlp3)
        
        # Define a final linear layer for classification
        self.linear = torch.nn.Linear(hidden_dim, 2)
    
    def forward(self, data):
        # First GIN layer
        x, edge_index = data.x.float(), data.edge_index.to(torch.int64)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        # Second GIN layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Third GIN layer
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        
        # Final linear layer
        x = self.linear(x)
        
        return x

class SiameseGNN_GIN(torch.nn.Module):
    def __init__(self, top_k, input_dim, dropout, nhidden):
        super(SiameseGNN_GIN, self).__init__()
        self.gnn = GIN(input_dim, nhidden)
        self.topk_layer = torch.topk
        self.top_k = top_k
        self.dropout = nn.Dropout(dropout)
        self.similarity = nn.PairwiseDistance()

        self.fc1 = Linear(top_k, 128)  # Adjust input size according to pooling output
        self.norm1 = LayerNorm(128)
        self.relu1 = ReLU()

        self.fc2 = Linear(128, nhidden)
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
