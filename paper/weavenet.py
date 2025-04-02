# # weavenet.py
#
# import torch
# import torch.nn as nn
#
#
# class EdgeUpdate(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super().__init__()
#         self.linear = nn.Sequential(
#             nn.Linear(input_dim, output_dim),
#             nn.ReLU()
#         )
#
#     def forward(self, src_x, dst_x, edge_attr):
#         concat = torch.cat([src_x, dst_x, edge_attr], dim=1)
#         return self.linear(concat)
#
#
# class NodeUpdate(nn.Module):
#     def __init__(self, node_input_dim, edge_input_dim, output_dim):
#         super().__init__()
#         self.linear = nn.Sequential(
#             nn.Linear(node_input_dim + edge_input_dim, output_dim),
#             nn.ReLU()
#         )
#
#     def forward(self, x, aggregated_edge_info):
#         concat = torch.cat([x, aggregated_edge_info], dim=1)
#         return self.linear(concat)
#
#
# class WeaveLayer(nn.Module):
#     def __init__(self, node_in, edge_in, node_out, edge_out):
#         super().__init__()
#         self.edge_update = EdgeUpdate(node_in * 2 + edge_in, edge_out)
#         self.node_update = NodeUpdate(node_in, edge_out, node_out)
#
#     def forward(self, x, edge_attr, edge_index):
#         src_x = x[edge_index[0]]
#         dst_x = x[edge_index[1]]
#
#         e_new = self.edge_update(src_x, dst_x, edge_attr)
#
#         agg = torch.zeros(x.size(0), e_new.size(1), device=x.device)
#         for i in range(e_new.size(0)):
#             target = edge_index[1, i].item()
#             agg[target] += e_new[i]
#
#         x_new = self.node_update(x, agg)
#         return x_new, e_new
#
#
# class WeaveNet(nn.Module):
#     def __init__(self, node_dim, edge_dim, hidden_dim, num_layers=2):
#         super().__init__()
#         self.layers = nn.ModuleList()
#
#         # First layer
#         self.layers.append(WeaveLayer(node_in=node_dim, edge_in=edge_dim,
#                                       node_out=hidden_dim, edge_out=hidden_dim))
#
#         # Subsequent layers: all hidden_dim in/out
#         for _ in range(1, num_layers):
#             self.layers.append(WeaveLayer(node_in=hidden_dim, edge_in=hidden_dim,
#                                           node_out=hidden_dim, edge_out=hidden_dim))
#
#         self.edge_final = nn.Linear(hidden_dim, 1)
#
#     def forward(self, data):
#         x = data.x
#         edge_attr = data.edge_attr
#         edge_index = data.edge_index
#
#         for layer in self.layers:
#             x, edge_attr = layer(x, edge_attr, edge_index)
#
#         scores = self.edge_final(edge_attr).squeeze(-1)
#         return scores
# Step 1: Improve WeaveNet architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class WeaveNet(nn.Module):
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
