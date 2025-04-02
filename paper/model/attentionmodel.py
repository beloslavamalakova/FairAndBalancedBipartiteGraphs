import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


# Custom bipartite attention layer between students and universities
class BipartiteAttentionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.student_proj = nn.Linear(input_dim, hidden_dim)
        self.college_proj = nn.Linear(input_dim, hidden_dim)
        self.att_score = nn.Linear(hidden_dim, 1)

    def forward(self, student_feats, college_feats):
        S = self.student_proj(student_feats).unsqueeze(1)  # (N, 1, H)
        C = self.college_proj(college_feats).unsqueeze(0)  # (1, M, H)
        scores = self.att_score(torch.tanh(S + C)).squeeze(-1)  # (N, M)
        weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(weights, college_feats)  # (N, input_dim)
        return attended, weights


# Sinkhorn normalization layer
class SinkhornNormalization(nn.Module):
    def __init__(self, num_iters=10):
        super().__init__()
        self.num_iters = num_iters

    def forward(self, scores):
        for _ in range(self.num_iters):
            scores = scores - torch.logsumexp(scores, dim=1, keepdim=True)
            scores = scores - torch.logsumexp(scores, dim=0, keepdim=True)
        return torch.exp(scores)


class FairMatchingGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, num_layers=6,
                 norm_type=None, loss_type='MSE'):
        super().__init__()
        self.proj = nn.Linear(node_dim, hidden_dim)

        self.gcn_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
            self.norm_layers.append(nn.LayerNorm(hidden_dim))

        self.attn_layer = BipartiteAttentionLayer(hidden_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.combined_proj = nn.Linear(3 * hidden_dim, 1)

        self.norm_type = norm_type
        if norm_type == 'sinkhorn':
            self.norm_layer = SinkhornNormalization()
        else:
            self.norm_layer = None

        self.loss_type = loss_type

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.proj(x))

        for gcn, norm in zip(self.gcn_layers, self.norm_layers):
            residual = x
            x = gcn(x, edge_index)
            x = F.relu(x + residual)
            x = norm(x)

        num_students = edge_index[0].max().item() + 1
        student_feats = x[:num_students]
        college_feats = x[num_students:]
        attn_student, attn_weights = self.attn_layer(student_feats, college_feats)

        edge_emb = self.edge_proj(edge_attr)
        edge_scores = []
        for i in range(edge_index.shape[1]):
            sid = edge_index[0, i].item()
            cid = edge_index[1, i].item() - num_students
            combined = torch.cat([
                attn_student[sid],
                college_feats[cid],
                edge_emb[i]
            ], dim=0)
            edge_scores.append(self.combined_proj(combined))
        scores = torch.stack(edge_scores).squeeze(-1)

        if self.norm_layer:
            num_students = edge_index[0].max().item() + 1
            num_colleges = edge_index[1].max().item() - num_students + 1

            # Sanity check: num_students * num_colleges should match score count
            expected_edges = num_students * num_colleges
            assert scores.shape[0] == expected_edges, (
                f"Expected {expected_edges} scores, got {scores.shape[0]}. "
                "Sinkhorn requires a full bipartite graph."
            )

            # Reshape to 2D (N x M), normalize, flatten back
            score_matrix = scores.view(num_students, num_colleges)
            score_matrix = self.norm_layer(score_matrix)
            scores = score_matrix.view(-1)

            return scores

    def compute_loss(self, scores, rank_matrix, edge_index, s_idx, c_idx):
        loss = 0.0
        for i in range(edge_index.shape[1]):
            sn = edge_index[0, i].item()
            cn = edge_index[1, i].item()
            inv_s = {v: k for k, v in s_idx.items()}
            inv_c = {v: k for k, v in c_idx.items()}
            s_id = inv_s.get(sn, None)
            c_id = inv_c.get(cn, None)
            if s_id is None or c_id is None:
                continue
            rank = rank_matrix.get(s_id, {}).get(c_id, 100)
            target = torch.tensor([float(rank)], dtype=torch.float, device=scores.device)
            pred = scores[i].unsqueeze(0)

            if self.loss_type == 'MSE':
                loss += F.mse_loss(pred, target)
            elif self.loss_type == 'L1':
                loss += F.l1_loss(pred, target)
            elif self.loss_type == 'SmoothL1':
                loss += F.smooth_l1_loss(pred, target)
            elif self.loss_type == 'Custom':
                raise ValueError(f"Still missing the equation: {self.loss_type}")
            else:
                raise ValueError(f"Unsupported loss type: {self.loss_type}")
        return loss
