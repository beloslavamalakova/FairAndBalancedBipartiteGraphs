# demo_dense.py

import torch
import random
import copy
from torch_geometric.data import Data

from data_structs import load_dense_dataset, FALLBACK_PREF_RANK
from classical_algs import gale_shapley_deferred_acceptance, serial_dictatorship, acda
from fairness_eval import compute_ef_k, exponential_score_difference, average_rank_difference, weighted_log_rank
from weavenet import WeaveNet
from train_experiment import simple_train_loop, hyperparam_experiment, build_match_from_scores


def convert_dense_students(students_raw, capacity=15):
    """
    students_raw: { sid -> [list_of_univ_in_pref_order] } (each is a list)
    Return: students_fixed dict => sid-> {preferences: {cid->rank}, matched_college: None, ...}
    """
    students_fixed = {}
    for sid, unis_list in students_raw.items():
        prefs_dict = {cid: rank for rank, cid in enumerate(unis_list)}
        students_fixed[sid] = {
            'preferences': prefs_dict,
            'matched_college': None,
            'current_proposal_index': 0,
            'capacity': capacity  # not used for students
        }
    return students_fixed


def convert_dense_colleges(colleges_raw, capacity=15):
    """
    colleges_raw: { cid -> [list_of_students_in_pref_order] }
    Return: colleges_fixed => cid -> { 'capacity':..., 'priority'(sid)->..., 'tentative_matches':[]}
    """
    colleges_fixed = {}
    for cid, stus_list in colleges_raw.items():
        ranking = {sid: rank for rank, sid in enumerate(stus_list)}
        def prio_func(sid):
            # higher is better => let's do negative rank, so lower rank => bigger prio
            return -ranking.get(sid, 9999999)
        colleges_fixed[cid] = {
            'capacity': capacity,
            'priority': prio_func,
            'tentative_matches': []
        }
    return colleges_fixed


def build_rank_matrix(students_fixed):
    """
    rank_matrix[sid][cid] = rank
    from each student's 'preferences'
    """
    rank_matrix = {}
    for sid, sdict in students_fixed.items():
        rank_matrix[sid] = {}
        for cid, rank_val in sdict['preferences'].items():
            rank_matrix[sid][cid] = rank_val
    return rank_matrix


def build_pyg_data(students_fixed, colleges_fixed, rank_matrix):
    s_ids = sorted(students_fixed.keys())
    c_ids = sorted(colleges_fixed.keys())
    s_idx = {sid: i for i, sid in enumerate(s_ids)}
    offset = len(s_ids)
    c_idx = {cid: offset + i for i, cid in enumerate(c_ids)}

    node_feats = torch.zeros(len(s_ids) + len(c_ids), 8)
    edges_src, edges_dst, edges_attr = [], [], []

    for sid in s_ids:
        for cid in c_ids:
            rank_val = rank_matrix[sid].get(cid, FALLBACK_PREF_RANK)
            edges_src.append(s_idx[sid])
            edges_dst.append(c_idx[cid])
            edges_attr.append([float(rank_val), random.random()])

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    edge_attr = torch.tensor(edges_attr, dtype=torch.float)
    data = Data(x=node_feats, edge_index=edge_index, edge_attr=edge_attr)
    return data, s_idx, c_idx


def evaluate_matching(label, match_dict, rank_matrix, priorities):
    assigned_ranks = []
    for sid, cid in match_dict.items():
        assigned_ranks.append(rank_matrix.get(sid, {}).get(cid, FALLBACK_PREF_RANK))

    esd_val = exponential_score_difference(assigned_ranks)
    ard_val = average_rank_difference(assigned_ranks)
    wlr_val = weighted_log_rank(assigned_ranks)

    print(f"\n{label.upper()} RESULTS:")
    print(f"ESD={esd_val:.3f}, ARD={ard_val:.3f}, WLR={wlr_val:.3f}")

    ef_k_val = compute_ef_k(rank_matrix, match_dict, priorities)
    print(f"{label.upper()} EF-k => k={ef_k_val}")


def main():
    # Load the raw JSON: { sid -> [list_of_univIDs in pref order] }, { cid -> [list_of_studentIDs] }
    students_raw, colleges_raw = load_dense_dataset()

    # Convert to fixed structures
    students_fixed = convert_dense_students(students_raw, capacity=9999)
    colleges_fixed = convert_dense_colleges(colleges_raw, capacity=15)

    # Build a rank_matrix for the GNN & fairness
    rank_matrix = build_rank_matrix(students_fixed)

    # Build priorities
    priorities = {cid: colleges_fixed[cid]['priority'] for cid in colleges_fixed}

    # Gale-Shapley
    gs_students = copy.deepcopy(students_fixed)
    gs_colleges = copy.deepcopy(colleges_fixed)
    gs_match = gale_shapley_deferred_acceptance(gs_students, gs_colleges)
    evaluate_matching("Gale-Shapley", gs_match, rank_matrix, priorities)

    # Serial Dictatorship
    sd_students = copy.deepcopy(students_fixed)
    sd_colleges = copy.deepcopy(colleges_fixed)
    ordering = list(sd_students.keys())
    random.shuffle(ordering)
    sd_match = serial_dictatorship(sd_students, sd_colleges, ordering)
    evaluate_matching("Serial Dictatorship", sd_match, rank_matrix, priorities)

    # ACDA
    acda_students = copy.deepcopy(students_fixed)
    acda_colleges = copy.deepcopy(colleges_fixed)
    art_caps = {cid: max(1, acda_colleges[cid]['capacity'] - 1) for cid in acda_colleges}
    acda_match = acda(acda_students, acda_colleges, art_caps)
    evaluate_matching("ACDA", acda_match, rank_matrix, priorities)

    # GNN
    data, s_idx, c_idx = build_pyg_data(students_fixed, colleges_fixed, rank_matrix)
    train_data_list = [(data, s_idx, c_idx, rank_matrix)]
    #model = WeaveNet(node_dim=8, edge_dim=2, hidden_dim=16, num_layers=2)
    model = WeaveNet(node_dim=8, edge_dim=2, hidden_dim=32, num_layers=4)


    print("\n--- SIMPLE TRAIN LOOP ---")
    simple_train_loop(model, train_data_list, epochs=10, lr=1e-3)
    model.eval()
    scores = model(data)
    gnn_match = build_match_from_scores(scores, data.edge_index, s_idx, c_idx, data)
    assigned_ranks_gnn = [
        rank_matrix.get(sid, {}).get(cid, FALLBACK_PREF_RANK)
        for sid, cid in gnn_match.items() if cid is not None
    ]
    print("GNN RESULTS:")
    esd_val = exponential_score_difference(assigned_ranks_gnn)
    ard_val = average_rank_difference(assigned_ranks_gnn)
    wlr_val = weighted_log_rank(assigned_ranks_gnn)
    print(f"ESD={esd_val:.3f}, ARD={ard_val:.3f}, WLR={wlr_val:.3f}")

    gnn_efk = compute_ef_k(rank_matrix, gnn_match, priorities)
    print(f"GNN EF-k => k={gnn_efk}")

    # Hyperparam
    print("\n--- HYPERPARAM EXPERIMENT ---")
    best_model, best_lr, best_val = hyperparam_experiment(
        WeaveNet,
        train_data_list,
        train_data_list,
        node_dim=8,
        edge_dim=2,
        hidden_dim=16,
        lr_candidates=[1e-3, 5e-4],
        epochs=5
    )
    print(f"Best LR: {best_lr}, best val ARD={best_val:.4f}")


if __name__ == "__main__":
    main()
