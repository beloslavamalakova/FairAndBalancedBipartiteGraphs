# data_structs.py
import os
import json
import random

DATA_DIR = "data"
FALLBACK_PREF_RANK = 100


def generate_dense_dataset(num_students=500, num_universities=50, capacity_per_university=15):
    """
    Creates dense_students.json: dict of {s_id -> [list_of_univ_in_pref_order]}
    Creates dense_universities.json: dict of {u_id -> [list_of_students_in_pref_order]}
    """
    students = {}
    colleges = {}

    for s_id in range(num_students):
        unis = list(range(num_universities))
        random.shuffle(unis)
        # store as a list
        students[s_id] = unis

    for u_id in range(num_universities):
        stus = list(range(num_students))
        random.shuffle(stus)
        colleges[u_id] = stus

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "dense_students.json"), "w") as f:
        json.dump(students, f)
    with open(os.path.join(DATA_DIR, "dense_universities.json"), "w") as f:
        json.dump(colleges, f)


def load_dense_dataset():
    """
    Loads dense_students.json and dense_universities.json
    Ensures keys are converted to int, returning:
      - students: { s_id -> [list_of_univ_in_pref_order] } (keys are int)
      - colleges: { u_id -> [list_of_students_in_pref_order] } (keys are int)
    """
    with open(os.path.join(DATA_DIR, "dense_students.json"), "r") as f:
        raw_students = json.load(f)
    with open(os.path.join(DATA_DIR, "dense_universities.json"), "r") as f:
        raw_colleges = json.load(f)

    # Convert keys to int
    students = {}
    for sid_str, unis_list in raw_students.items():
        sid = int(sid_str)
        # unis_list is a list of int
        students[sid] = unis_list

    colleges = {}
    for cid_str, stus_list in raw_colleges.items():
        cid = int(cid_str)
        # stus_list is a list of int
        colleges[cid] = stus_list

    return students, colleges


def generate_sparse_dataset(num_students=500, num_universities=50, capacity_per_university=15, e=0.3):
    """
    Creates sparse_students.json: dict of { sid -> partialranking + fallback? }
    Creates sparse_universities.json: dict of { uid -> partialranking + fallback? }
    """
    # In your final usage, you might override capacity_per_university if needed.
    # For now, we just store partial lists and fill with fallback as needed.
    students = {}
    colleges = {}

    # each student ranks at least e * unis
    for s_id in range(num_students):
        unis_count = max(1, int(e * num_universities))
        chosen = random.sample(range(num_universities), unis_count)
        # store as an entire list from 0..(num_unis-1) with fallback
        # but simpler: chosen in random order => they get real rank, the rest => fallback
        # We'll do a list: index = rank, value = c_id => or do a dict for stable marriage?
        # For your final code, let's store them as a list in random order.
        # Then others have fallback or huge rank in the next layer of code usage.

        # For now let's just store the partial preference as a dict {cid: rank}, fallback is set later
        pref_dict = {}
        for r, c_id in enumerate(chosen):
            pref_dict[c_id] = r
        students[s_id] = pref_dict

    for u_id in range(num_universities):
        stus_count = max(1, int(e * num_students))
        chosen_stus = random.sample(range(num_students), stus_count)
        pref_dict = {}
        for r, s_id in enumerate(chosen_stus):
            pref_dict[s_id] = r
        colleges[u_id] = pref_dict

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "sparse_students.json"), "w") as f:
        json.dump(students, f)
    with open(os.path.join(DATA_DIR, "sparse_universities.json"), "w") as f:
        json.dump(colleges, f)


def load_sparse_dataset():
    """
    loads sparse_students.json and sparse_universities.json
    keys => int, values => partial dict
    """
    with open(os.path.join(DATA_DIR, "sparse_students.json"), "r") as f:
        raw_s = json.load(f)
    with open(os.path.join(DATA_DIR, "sparse_universities.json"), "r") as f:
        raw_u = json.load(f)

    students = {}
    for sid_str, pref_dict in raw_s.items():
        sid = int(sid_str)
        # ensure subkeys are int too
        new_pref = {}
        for cid_str, rank_val in pref_dict.items():
            new_pref[int(cid_str)] = rank_val
        students[sid] = new_pref

    colleges = {}
    for cid_str, pref_dict in raw_u.items():
        cid = int(cid_str)
        new_pref = {}
        for s_str, r_val in pref_dict.items():
            new_pref[int(s_str)] = r_val
        colleges[cid] = new_pref

    return students, colleges

generate_dense_dataset()
generate_sparse_dataset()
