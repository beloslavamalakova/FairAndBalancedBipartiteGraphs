# fairness_eval.py
import math

FALLBACK_PREF_RANK = 10**6

def get_student_pref_index(student_id, college_id, rank_matrix):
    return rank_matrix.get(student_id, {}).get(college_id, FALLBACK_PREF_RANK)

def compute_ef_k(rank_matrix, matching, priorities):
    envy_counts = {sid: 0 for sid in rank_matrix}

    inv_matches = {}
    for sid, cid in matching.items():
        if cid is not None:
            inv_matches.setdefault(cid, []).append(sid)

    for sid in rank_matrix:
        c_assigned = matching.get(sid, None)
        if c_assigned is None:
            continue
        for sid2 in rank_matrix:
            if sid == sid2:
                continue
            c2 = matching.get(sid2, None)
            if c2 is None:
                continue
            if get_student_pref_index(sid, c2, rank_matrix) < get_student_pref_index(sid, c_assigned, rank_matrix):
                if priorities[c2](sid) > priorities[c2](sid2):
                    envy_counts[sid] += 1

    return max(envy_counts.values()) if envy_counts else 0

def exponential_score_difference(ranks):
    if not ranks:
        return 0.0
    gap = max(ranks) - min(ranks)
    return math.exp(gap) - 1.0

def average_rank_difference(ranks):
    n = len(ranks)
    if n <= 1:
        return 0.0
    total_diff = 0
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            total_diff += abs(ranks[i] - ranks[j])
            count += 1
    return total_diff / count

def weighted_log_rank(ranks):
    if not ranks:
        return 0.0
    r_min = min(ranks)
    r_max = max(ranks)
    val = 0.0
    if r_min > 1:
        val += r_min * math.log(r_min)
    if r_max > 1:
        val += r_max * math.log(r_max)
    val += abs(r_max - r_min)
    return val
