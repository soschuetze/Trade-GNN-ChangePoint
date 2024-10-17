import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv, SAGEConv, global_sort_pool
from torch_geometric.nn.aggr import SortAggregation
from torch.nn import Linear, LayerNorm, ReLU, Sigmoid, BatchNorm1d
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
import torch_geometric as pyg

class GCNEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = SAGEConv(in_channels=27, out_channels=128)
        self.conv2 = SAGEConv(in_channels=128, out_channels=64)

    def forward(self, g):
        x = self.conv1(g.x, g.edge_index.to(torch.int64))
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.dropout(x, 0.1)
        x = self.conv2(x, g.edge_index.to(torch.int64))
        return x 
    
    
class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = GCNEncoder()
        self.pairwise_distance = nn.PairwiseDistance()
        self.pooling = pyg.nn.pool.TopKPooling(in_channels=1, ratio=64)
        self.linear1 = nn.Linear(64, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.linear2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.linear3 = nn.Linear(16, 1)

    def forward(self, g1, g2):
        x1 = self.encoder(g1)
        x2 = self.encoder(g2)
        x = self.pairwise_distance(x1, x2).unsqueeze(1)
        x, *_ = self.pooling(
            x=x, 
            edge_index=g1.edge_index, 
            batch=g1.batch # this ensures top-k is selected per graph
        ) 
        x = x.view(-1, 64) # the top-k will be concatenated, this undoes this into a batch dimension 
        x = self.linear1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = torch.nn.functional.dropout(x, 0.1)
        x = self.linear2(x)
        x = self.bn2(x)
        x = nn.functional.relu(x)
        x = torch.nn.functional.dropout(x, 0.1)
        x = self.linear3(x)
        x = torch.sigmoid(x)
        return x