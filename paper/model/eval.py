import torch
import torch.nn.functional as F
from utils import build_match_from_scores
from paper.fairness_eval import (
    compute_ef_k, exponential_score_difference,
    average_rank_difference, weighted_log_rank
)

def evaluate_model(model, data_list, s_idx, c_idx, rank_matrix, students, priorities, capacities=None):
    model.eval()
    with torch.no_grad():
        data = data_list  # already a Data object
        #data = data_list[0][0]  # assuming one graph for now
        scores = model(data)

    match_dict = build_match_from_scores(
        scores, data.edge_index, s_idx, c_idx, data, capacities
    )

    assigned_ranks = [
        rank_matrix.get(sid, {}).get(cid, 100)
        for sid, cid in match_dict.items()
        if cid is not None
    ]

    print("\nGNN RESULTS:")
    print(f"ESD={exponential_score_difference(assigned_ranks):.3f}, "
          f"ARD={average_rank_difference(assigned_ranks):.3f}, "
          f"WLR={weighted_log_rank(assigned_ranks):.3f}")
    ef_k = compute_ef_k(rank_matrix, match_dict, priorities)
    print(f"GNN EF-k => k={ef_k}")

    return match_dict, assigned_ranks, ef_k


def make_priority_functions(colleges):
    priority_fns = {}
    for cid, pref in colleges.items():
        if isinstance(pref, list):
            # Convert list to {student_id: rank}
            pref_dict = {sid: rank for rank, sid in enumerate(pref)}
        else:
            pref_dict = pref  # Already a dict in sparse dataset

        priority_fns[cid] = lambda sid, d=pref_dict: d.get(sid, 10**6)
    return priority_fns

