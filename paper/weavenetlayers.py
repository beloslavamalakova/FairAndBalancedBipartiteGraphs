# weavenetlayers.py
# Step 1: Improve WeaveNet architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

class ImprovedWeaveNet(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=64, num_layers=4):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)

        self.layers = nn.ModuleList([
            WeaveLayer(hidden_dim) for _ in range(num_layers)
        ])

        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        x = self.node_proj(data.x)
        edge_attr = self.edge_proj(data.edge_attr)

        for layer in self.layers:
            x = layer(x, data.edge_index, edge_attr)

        src, dst = data.edge_index
        edge_repr = torch.cat([x[src], x[dst]], dim=1)
        scores = self.final_mlp(edge_repr).squeeze(-1)
        return scores


class WeaveLayer(MessagePassing):
    def __init__(self, hidden_dim):
        super().__init__(aggr='add')
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        return self.mlp(torch.cat([x_i + x_j, edge_attr], dim=1))

    def update(self, aggr_out):
        return aggr_out
