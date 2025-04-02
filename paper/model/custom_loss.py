import torch
import torch.nn.functional as F

def compute_custom_loss(scores, edge_index, s_idx, c_idx, rank_matrix,
                        lambda_pred=1.0,
                        lambda_avg_s=0.5, lambda_avg_c=0.5,
                        fallback_value=100):
    pred_loss = 0.0
    student_scores = {}
    college_scores = {}

    inv_s = {v: k for k, v in s_idx.items()}
    inv_c = {v: k for k, v in c_idx.items()}

    for i in range(edge_index.shape[1]):
        sn = edge_index[0, i].item()
        cn = edge_index[1, i].item()

        s_id = inv_s.get(sn)
        c_id = inv_c.get(cn)

        if s_id is None or c_id is None:
            continue

        # Rank target from data
        target_rank = rank_matrix.get(s_id, {}).get(c_id, fallback_value)
        target = torch.tensor([float(target_rank)], dtype=torch.float, device=scores.device)
        pred = scores[i].unsqueeze(0)

        # Prediction loss
        pred_loss += F.mse_loss(pred, target)

        # Collect predictions
        student_scores.setdefault(s_id, []).append(pred)
        college_scores.setdefault(c_id, []).append(pred)

    # Average satisfaction losses
    if student_scores:
        avg_s = torch.stack([torch.mean(torch.stack(v)) for v in student_scores.values()]).mean()
    else:
        avg_s = torch.tensor(0.0, device=scores.device)

    if college_scores:
        avg_c = torch.stack([torch.mean(torch.stack(v)) for v in college_scores.values()]).mean()
    else:
        avg_c = torch.tensor(0.0, device=scores.device)

    # Final loss
    total_loss = (
        lambda_pred * pred_loss +
        lambda_avg_s * avg_s +
        lambda_avg_c * avg_c
    )

    return total_loss
