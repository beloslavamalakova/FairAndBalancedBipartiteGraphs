import torch
import torch.nn.functional as F


def simple_train_loop(model, train_data_list, epochs=10, lr=1e-3,
                      opt_type='AdamW', loss_type='SmoothL1', fallback=100):
    """
    Train a model using student-college preference ranks.

    :param model: The GNN model to train
    :param train_data_list: List of (data, s_idx, c_idx, rank_matrix) tuples
    :param epochs: Number of epochs to train
    :param lr: Learning rate
    :param opt_type: Optimizer type: 'Adam', 'AdamW', 'RMSprop', 'SGD'
    :param loss_type: Loss type: 'MSE', 'L1', 'SmoothL1'
    :param fallback: The rank to skip during training (e.g., unranked pairs)
    """

    # Select optimizer
    if opt_type == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif opt_type == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    elif opt_type == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    elif opt_type == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown opt_type: {opt_type}")

    # Select loss function
    if loss_type == 'MSE':
        loss_fn = F.mse_loss
    elif loss_type == 'L1':
        loss_fn = F.l1_loss
    elif loss_type == 'SmoothL1':
        loss_fn = F.smooth_l1_loss
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for data, s_idx, c_idx, rank_matrix in train_data_list:
            optimizer.zero_grad()
            scores = model(data)
            edge_index = data.edge_index
            loss = 0.0

            inv_s = {v: k for k, v in s_idx.items()}
            inv_c = {v: k for k, v in c_idx.items()}

            for i in range(edge_index.shape[1]):
                sn = edge_index[0, i].item()
                cn = edge_index[1, i].item()
                sid = inv_s.get(sn)
                cid = inv_c.get(cn)

                if sid is None or cid is None:
                    continue

                rank = rank_matrix.get(sid, {}).get(cid, fallback)
                if rank == fallback:
                    continue  # skip unranked

                target = torch.tensor([float(rank)], dtype=torch.float)
                pred = scores[i].unsqueeze(0)
                loss += loss_fn(pred, target)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, 'loss' {total_loss:.4f}")




def build_match_from_scores(scores, edge_index, s_idx, c_idx, data, capacities=None):
    edges_list = []
    scores_np = scores.detach().cpu().numpy()
    for i in range(scores_np.shape[0]):
        sn = edge_index[0, i].item()
        cn = edge_index[1, i].item()
        sc = scores_np[i]
        edges_list.append((sn, cn, sc))
    edges_list.sort(key=lambda x: x[2], reverse=True)

    matched_students = set()
    matched_college_count = {}
    matching = {}

    inv_sidx = {v: k for k, v in s_idx.items()}
    inv_cidx = {v: k for k, v in c_idx.items()}

    for (sn, cn, sc) in edges_list:
        if sn in matched_students:
            continue
        cap_used = matched_college_count.get(cn, 0)
        cap_limit = capacities.get(cn, 1) if capacities else 15

        if cap_used < cap_limit:
            matched_students.add(sn)
            matched_college_count[cn] = cap_used + 1

            if sn in inv_sidx and cn in inv_cidx:
                s_id = inv_sidx[sn]
                c_id = inv_cidx[cn]
                matching[s_id] = c_id
    return matching

def hyperparam_experiment(model_class, train_data_list, val_data_list,
                          node_dim, edge_dim, hidden_dim, lr_candidates, epochs=5):
    best_val = float('inf')
    best_model = None
    best_lr = None
    for lr in lr_candidates:
        model = model_class(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim, num_layers=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        model.train()
        for epoch in range(epochs):
            for data, s_idx, c_idx, rank_matrix in train_data_list:
                optimizer.zero_grad()
                scores = model(data)
                edge_index = data.edge_index
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
                    rank = rank_matrix.get(s_id, {}).get(c_id, 100.0)
                    norm_rank = float(rank) / 100.0
                    loss += F.mse_loss(scores[i].unsqueeze(0), torch.tensor([norm_rank], dtype=torch.float))
                loss.backward()
                optimizer.step()
        # Evaluate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, s_idx, c_idx, rank_matrix in val_data_list:
                scores = model(data)
                edge_index = data.edge_index
                for i in range(edge_index.shape[1]):
                    sn = edge_index[0, i].item()
                    cn = edge_index[1, i].item()
                    inv_s = {v: k for k, v in s_idx.items()}
                    inv_c = {v: k for k, v in c_idx.items()}
                    s_id = inv_s.get(sn, None)
                    c_id = inv_c.get(cn, None)
                    if s_id is None or c_id is None:
                        continue
                    rank = rank_matrix.get(s_id, {}).get(c_id, 100.0)
                    norm_rank = float(rank) / 100.0
                    val_loss += abs(scores[i].item() - norm_rank)
        avg_val_loss = val_loss / len(val_data_list)
        print(f"LR={lr}, val ARD={avg_val_loss:.4f}")
        if avg_val_loss < best_val:
            best_val = avg_val_loss
            best_model = model
            best_lr = lr
    print(f"Best LR: {best_lr}, best val ARD={best_val:.4f}")
    return best_model, best_lr, best_val
