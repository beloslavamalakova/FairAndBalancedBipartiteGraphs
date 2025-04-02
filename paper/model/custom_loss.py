import torch
import torch.nn.functional as F


def compute_custom_loss(scores, edge_index, s_idx, c_idx, rank_matrix,
                        lambda_pred=1.0, lambda_wlr=1.0,
                        lambda_avg_s=0.5, lambda_avg_c=0.5,
                        fallback_value=100):
    total_loss = 0.0
    pred_loss = 0.0
    #efk_loss = 0.0
    wlr_loss = 0.0
    student_ranks = {}
    college_ranks = {}
    inv_s = {v: k for k, v in s_idx.items()}
    inv_c = {v: k for k, v in c_idx.items()}

    for i in range(edge_index.shape[1]):
        sn = edge_index[0, i].item()
        cn = edge_index[1, i].item()

        s_id = inv_s.get(sn)
        c_id = inv_c.get(cn)

        if s_id is None or c_id is None:
            continue

        rank = rank_matrix.get(s_id, {}).get(c_id, fallback_value)
        pred = scores[i].unsqueeze(0)
        target = torch.tensor([float(rank)], dtype=torch.float, device=pred.device)

        # 1. Prediction Loss
        pred_loss += F.mse_loss(pred, target)

        # 2. Accumulate ranks for balance losses
        student_ranks.setdefault(s_id, []).append(rank)
        college_ranks.setdefault(c_id, []).append(rank)

        # 3. EF-k soft violation (if any student's rank worse than other's envy)
        # for other_sn in range(edge_index.shape[1]):
        #     if other_sn == i:
        #         continue
        #     other_cn = edge_index[1, other_sn].item()
        #     other_s_id = inv_s.get(edge_index[0, other_sn].item())
        #     other_c_id = inv_c.get(other_cn)
        #     if other_s_id is None or other_c_id is None:
        #         continue
        #
        #     rank_i = rank_matrix.get(s_id, {}).get(c_id, fallback_value)
        #     rank_j = rank_matrix.get(s_id, {}).get(other_c_id, fallback_value)
        #     rank_jj = rank_matrix.get(other_s_id, {}).get(other_c_id, fallback_value)
        #
        #     if rank_j < rank_i and rank_jj > 0:  # envy + other doesn't value
        #         efk_loss += (rank_i - rank_j) / fallback_value  # normalize to [0,1]

        # 4. WLR loss: penalize high ranks with log function
        if rank > 0:
            wlr_loss += rank * torch.log(torch.tensor(rank, dtype=torch.float, device=pred.device))

    # 5. Average satisfaction loss for balance
    avg_s = torch.tensor([sum(v)/len(v) for v in student_ranks.values()]).mean()
    avg_c = torch.tensor([sum(v)/len(v) for v in college_ranks.values()]).mean()

    total_loss = (
        lambda_pred * pred_loss +
        lambda_wlr * wlr_loss +
        lambda_avg_s * avg_s +
        lambda_avg_c * avg_c
    )

    return total_loss
