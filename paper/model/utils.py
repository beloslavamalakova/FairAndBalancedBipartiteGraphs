# match_utils.py

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
