import sys
from torch_geometric.nn import global_mean_pool, global_max_pool
import torch.nn as nn
import torch
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_geometric.nn.pool.select import SelectTopK
from embedding import MLP

class GraphSiamese(nn.Module):
    """Architecture to detect if two graphs are different. """

    def __init__(self, embedding: nn.Module, similarity: str, pooling: str, loss: str, top_k: int, nlinear=2, nhidden=16,
                 dropout=0.1,
                 features=None):
        super(GraphSiamese, self).__init__()
        self.embedding = embedding
        self.input_dim = embedding.input_dim
        self.features = features

        self.similarity = nn.PairwiseDistance()
        if pooling == 'topk':
            self.descending = True
        self.pooling = pooling

        self.pooling_layer = torch.topk
        self.top_k = top_k
        self.nlinear = nlinear
        self.mlp = MLP(nlinear, top_k, nhidden, 1, dropout=dropout)
        self.loss = loss

    def forward(self, data1, data2):

        graph1_encoding = self.embedding(data1)
        graph2_encoding = self.embedding(data2)

        num_nodes = data1.num_nodes

        if self.pooling == 'avgraph':  
            graph1_encoding = self.pooling_layer(graph1_encoding, data1.batch)
            graph2_encoding = self.pooling_layer(graph2_encoding, data2.batch)

        similarity = self.similarity(graph1_encoding.squeeze(), graph2_encoding.squeeze()).unsqueeze(1)
        similarity = similarity.view(6, 199)

        x, top_indices = torch.topk(similarity, self.top_k, dim=1)

        x = x.squeeze()
        if self.nlinear == 0:  
            x = torch.nn.AvgPool1d(x)
        else:  
            if x.dim() < 2:
                x = torch.unsqueeze(x, 0)
            x = self.mlp(x)

        return x