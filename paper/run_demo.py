import torch
from torch_geometric.data import Data
import random
import copy

from data_structs import (
    generate_synthetic_data,
    generate_sparse_preference_data,
    STUDENT_PREFERENCES
)
from classical_algs import gale_shapley_deferred_acceptance, serial_dictatorship
from fairness_eval import (
    compute_ef_k, exponential_score_difference,
    average_rank_difference, weighted_log_rank
)
from weavenet import WeaveNet
from train_experiment import (
    simple_train_loop,
    hyperparam_experiment,
    build_match_from_scores
)

USE_SPARSE = True  # Toggle between True (sparse) or False (dense)
FALLBACK_PREF_RANK = 10**6

def build_pyg_data(students, colleges):
    s_idx = {}
    c_idx = {}
    for i, s in enumerate(students):
        s_idx[s['id']] = i
    base = len(students)
    for j, c_id in enumerate(colleges):
        c_idx[c_id] = base + j

    node_count = len(students) + len(colleges)
    node_feats = torch.zeros(node_count, 8)

    edges_src = []
    edges_dst = []
    edges_attr = []

    for s in students:
        sid = s['id']
        for c_id in colleges:
            pref_rank = STUDENT_PREFERENCES.get(sid, {}).get(c_id, FALLBACK_PREF_RANK)
            prio = random.random()

            src_n = s_idx[sid]
            dst_n = c_idx[c_id]
            edges_src.append(src_n)
            edges_dst.append(dst_n)
            edges_attr.append([float(pref_rank), float(prio)])

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    edge_attr = torch.tensor(edges_attr, dtype=torch.float)
    data = Data(x=node_feats, edge_index=edge_index, edge_attr=edge_attr)
    return data, s_idx, c_idx

def evaluate_matching(label, match_dict, students, rank_matrix, priorities, fallback=True):
    assigned_ranks = []
    num_colleges = max((len(ranks) for ranks in rank_matrix.values()), default=0)

    for sid, cid in match_dict.items():
        if cid is None or sid not in rank_matrix or cid not in rank_matrix[sid]:
            if fallback:
                assigned_ranks.append(FALLBACK_PREF_RANK)
            else:
                continue  # Skip unmatched
        else:
            assigned_ranks.append(rank_matrix[sid][cid])

    print(f"\n{label.upper()} RESULTS:")
    print(f"ESD={exponential_score_difference(assigned_ranks):.3f}, "
          f"ARD={average_rank_difference(assigned_ranks):.3f}, "
          f"WLR={weighted_log_rank(assigned_ranks):.3f}")
    ef_k = compute_ef_k(students, match_dict, priorities)
    print(f"{label.upper()} EF-k => k={ef_k}")

def main_demo():
    if USE_SPARSE:
        students, colleges = generate_sparse_preference_data(num_students=100, num_colleges=10, capacity_per_college=12, e=0.3)
    else:
        students, colleges = generate_synthetic_data(num_students=100, num_colleges=10, capacity_per_college=12)

    rank_matrix = {
        s['id']: {
            cid: STUDENT_PREFERENCES[s['id']][cid]
            for cid in STUDENT_PREFERENCES[s['id']]
        }
        for s in students
        if s['id'] in STUDENT_PREFERENCES
    }

    # patch: ensure rank_matrix has full mapping even if fallback is needed
    for s in students:
        sid = s['id']
        if sid not in rank_matrix:
            rank_matrix[sid] = {}
        for cid in colleges:
            if cid not in rank_matrix[sid]:
                rank_matrix[sid][cid] = FALLBACK_PREF_RANK

    priorities = {cid: (lambda s: s) for cid in colleges}

    fallback = USE_SPARSE  # Only use fallback rank in sparse mode

    # Gale-Shapley
    gs_students = copy.deepcopy(students)
    gs_colleges = copy.deepcopy(colleges)
    gs_match = gale_shapley_deferred_acceptance(gs_students, gs_colleges)
    evaluate_matching("Gale-Shapley", gs_match, gs_students, rank_matrix, priorities, fallback=fallback)

    # Serial Dictatorship
    sd_students = copy.deepcopy(students)
    sd_colleges = copy.deepcopy(colleges)
    ordering = list(range(len(sd_students)))
    random.shuffle(ordering)
    sd_match = serial_dictatorship(sd_students, sd_colleges, ordering)
    evaluate_matching("Serial Dictatorship", sd_match, sd_students, rank_matrix, priorities, fallback=fallback)

    # ACDA
    acda_students = copy.deepcopy(students)
    acda_colleges = copy.deepcopy(colleges)
    for cid in acda_colleges:
        acda_colleges[cid]['capacity'] = max(1, acda_colleges[cid]['capacity'] - 1)
    acda_match = gale_shapley_deferred_acceptance(acda_students, acda_colleges)
    evaluate_matching("ACDA", acda_match, acda_students, rank_matrix, priorities, fallback=fallback)

    # GNN setup
    data, s_idx, c_idx = build_pyg_data(students, colleges)
    train_data_list = [(data, s_idx, c_idx, rank_matrix)]
    val_data_list = [(data, s_idx, c_idx, rank_matrix)]
    model = WeaveNet(node_dim=8, edge_dim=2, hidden_dim=16, num_layers=2)

    print("\n--- SIMPLE TRAIN LOOP ---")
    simple_train_loop(model, train_data_list, epochs=5, lr=1e-3)

    model.eval()
    scores = model(data)
    gnn_match = build_match_from_scores(scores, data.edge_index, s_idx, c_idx, data)
    num_colleges = len(c_idx)
    assigned_ranks_gnn = [
        rank_matrix.get(sid, {}).get(cid, FALLBACK_PREF_RANK if fallback else 0)
        for sid, cid in gnn_match.items()
        if cid is not None and sid in rank_matrix and (fallback or cid in rank_matrix[sid])
    ]
    print("GNN MATCH RESULTS:")
    print(f"ESD={exponential_score_difference(assigned_ranks_gnn):.3f}, "
          f"ARD={average_rank_difference(assigned_ranks_gnn):.3f}, "
          f"WLR={weighted_log_rank(assigned_ranks_gnn):.3f}")
    gnn_efk = compute_ef_k(students, gnn_match, priorities)
    print(f"GNN EF-k => k={gnn_efk}")

    print("\n--- HYPERPARAM EXPERIMENT ---")
    best_model, best_lr, best_val = hyperparam_experiment(
        WeaveNet,
        train_data_list,
        val_data_list,
        node_dim=8,
        edge_dim=2,
        hidden_dim=16,
        lr_candidates=[1e-3, 5e-4],
        epochs=5
    )
    print(f"Best LR: {best_lr}, best val ARD={best_val:.4f}")

if __name__ == "__main__":
    main_demo()
