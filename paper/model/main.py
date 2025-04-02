import torch
from attentionmodel import FairMatchingGNN
from train import train
from eval import evaluate_model, make_priority_functions
from paper.data_structs import load_dense_dataset, load_sparse_dataset, FALLBACK_PREF_RANK
from custom_loss import compute_custom_loss
from torch_geometric.data import Data
import random

# Configuration
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

EPOCHS = 25
LR = 1e-3
OPT_TYPE = "AdamW"
LOSS_TYPE = "SmoothL1"  # This is still used internally in the model for prediction
HIDDEN_DIM = 64
NUM_LAYERS = 6
USE_SPARSE = True  # Toggle between dense and sparse

# Lambda weights for custom loss
LAMBDA_PRED = 0.1
LAMBDA_EFK = 2.0
LAMBDA_WLR = 2.0
LAMBDA_AVG_S = 1.0
LAMBDA_AVG_C = 1.0

def build_pyg_data_dense(students, colleges):
    s_ids = list(students.keys())
    c_ids = list(colleges.keys())

    s_idx = {sid: i for i, sid in enumerate(s_ids)}
    offset = len(s_ids)
    c_idx = {cid: offset + i for i, cid in enumerate(c_ids)}

    node_feats = torch.zeros(len(s_ids) + len(c_ids), 8)
    edges_src, edges_dst, edges_attr = [], [], []
    rank_matrix = {}

    for sid in s_ids:
        rank_matrix[sid] = {}
        for rank, cid in enumerate(students[sid]):
            edges_src.append(s_idx[sid])
            edges_dst.append(c_idx[cid])
            edges_attr.append([float(rank), random.random()])
            rank_matrix[sid][cid] = rank

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    edge_attr = torch.tensor(edges_attr, dtype=torch.float)
    data = Data(x=node_feats, edge_index=edge_index, edge_attr=edge_attr)
    return data, s_idx, c_idx, rank_matrix

def build_pyg_data_sparse(students, colleges):
    s_ids = list(students.keys())
    c_ids = list(colleges.keys())

    s_idx = {sid: i for i, sid in enumerate(s_ids)}
    offset = len(s_ids)
    c_idx = {cid: offset + i for i, cid in enumerate(c_ids)}

    node_feats = torch.zeros(len(s_ids) + len(c_ids), 8)
    edges_src, edges_dst, edges_attr = [], [], []
    rank_matrix = {}

    for sid in s_ids:
        rank_matrix[sid] = {}
        for cid in range(len(c_ids)):
            rank = students[sid].get(cid, FALLBACK_PREF_RANK)
            rank_matrix[sid][cid] = rank
            if rank != FALLBACK_PREF_RANK:
                edges_src.append(s_idx[sid])
                edges_dst.append(c_idx[cid])
                edges_attr.append([float(rank), random.random()])

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    edge_attr = torch.tensor(edges_attr, dtype=torch.float)
    data = Data(x=node_feats, edge_index=edge_index, edge_attr=edge_attr)
    return data, s_idx, c_idx, rank_matrix

def main():
    if USE_SPARSE:
        students, colleges = load_sparse_dataset()
        data, s_idx, c_idx, rank_matrix = build_pyg_data_sparse(students, colleges)
    else:
        students, colleges = load_dense_dataset()
        data, s_idx, c_idx, rank_matrix = build_pyg_data_dense(students, colleges)

    data = data.to(DEVICE)

    # Initialize model
    model = FairMatchingGNN(
        node_dim=data.x.shape[1],
        edge_dim=data.edge_attr.shape[1],
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        loss_type=LOSS_TYPE
    ).to(DEVICE)

    # Priorities for EF-k evaluation
    priorities = make_priority_functions(colleges)

    # Initialize loss
    # Train
    train(
        model, data, s_idx, c_idx, rank_matrix,
        lr=LR, epochs=EPOCHS, opt_type=OPT_TYPE,
        lambda_pred=LAMBDA_PRED,
        lambda_efk=LAMBDA_EFK,
        lambda_wlr=LAMBDA_WLR,
        lambda_avg_s=LAMBDA_AVG_S,
        lambda_avg_c=LAMBDA_AVG_C
    )


    # Evaluate
    evaluate_model(model, data, s_idx, c_idx, rank_matrix, students, priorities)

if __name__ == '__main__':
    main()
