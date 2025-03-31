# classical_algs.py

def gale_shapley_deferred_acceptance(students, colleges):
    """
    students: each is a dict { 'preferences': {cid->rank}, 'matched_college':None, 'current_proposal_index':0 }
              But we might store them as a list or dict. We'll unify to a dict of dict approach:
                { sid -> { 'preferences':..., 'matched_college':..., 'current_proposal_index':... } }
    colleges: each is a dict { 'capacity':..., 'priority':..., 'tentative_matches': list of sids }
    returns match_dict: { sid -> cid }
    """
    if isinstance(students, list):
        # convert to dict
        s_dict = {}
        for s in students:
            sid = s.get('id')
            s_dict[sid] = s
        students = s_dict

    unmatched = list(students.keys())
    while unmatched:
        s_id = unmatched.pop(0)
        s_obj = students[s_id]
        if 'current_proposal_index' not in s_obj:
            s_obj['current_proposal_index'] = 0
        if s_obj['current_proposal_index'] >= len(s_obj['preferences']):
            continue
        # find next college by ascending rank
        # but s_obj['preferences'] is a dict {cid->rank} or sorted list?
        # If it's dict, we need a sorted list of cids by rank. We'll do:
        sorted_prefs = sorted(s_obj['preferences'].items(), key=lambda x: x[1])
        if s_obj['current_proposal_index'] >= len(sorted_prefs):
            continue
        c_id = sorted_prefs[s_obj['current_proposal_index']][0]
        s_obj['current_proposal_index'] += 1

        c_obj = colleges[c_id]
        if len(c_obj['tentative_matches']) < c_obj['capacity']:
            c_obj['tentative_matches'].append(s_id)
            s_obj['matched_college'] = c_id
        else:
            worst_current = min(c_obj['tentative_matches'], key=lambda x: c_obj['priority'](x))
            if c_obj['priority'](s_id) > c_obj['priority'](worst_current):
                c_obj['tentative_matches'].remove(worst_current)
                c_obj['tentative_matches'].append(s_id)
                students[worst_current]['matched_college'] = None
                unmatched.append(worst_current)
                s_obj['matched_college'] = c_id
            else:
                unmatched.append(s_id)

    return {s_id: s_obj.get('matched_college') for s_id, s_obj in students.items()}


def serial_dictatorship(students, colleges, ordering):
    """
    students: dict { sid -> { 'preferences':..., 'matched_college':...} }
    ordering: list of sids in pick order
    """
    if isinstance(students, list):
        # convert
        s_dict = {}
        for s in students:
            sid = s['id']
            s_dict[sid] = s
        students = s_dict

    for s_id in ordering:
        s_obj = students[s_id]
        # get sorted list of (cid->rank) by rank
        sorted_prefs = sorted(s_obj['preferences'].items(), key=lambda x: x[1])
        for c_id, _ in sorted_prefs:
            if len(colleges[c_id]['tentative_matches']) < colleges[c_id]['capacity']:
                colleges[c_id]['tentative_matches'].append(s_id)
                s_obj['matched_college'] = c_id
                break

    return {sid: students[sid].get('matched_college') for sid in students}


def acda(students, colleges, artificial_caps):
    """
    Adjust capacities then run gale_shapley_deferred_acceptance,
    restore capacities
    """
    backup_caps = {}
    for c_id in colleges:
        backup_caps[c_id] = colleges[c_id]['capacity']
        colleges[c_id]['capacity'] = artificial_caps.get(c_id, backup_caps[c_id])

    match = gale_shapley_deferred_acceptance(students, colleges)

    for c_id in colleges:
        colleges[c_id]['capacity'] = backup_caps[c_id]

    return match
