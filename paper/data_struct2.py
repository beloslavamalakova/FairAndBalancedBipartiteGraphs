#data_struct2.py
import random

# Global student preference dictionary
STUDENT_PREFERENCES = {}

# Global college preference dictionary
COLLEGE_PREFERENCES = {}

def generate_sparse_preference_data(num_students=100, num_colleges=10, capacity_per_college=10, e=0.3):
    """
    Generate sparse student-college preference data.
    Each student ranks at least e * num_colleges colleges.
    Each college ranks at least e * num_students students.
    Only mutual rankings are kept.
    """
    global STUDENT_PREFERENCES, COLLEGE_PREFERENCES
    STUDENT_PREFERENCES = {}
    COLLEGE_PREFERENCES = {}

    students = []
    colleges = {}

    student_ids = list(range(num_students))
    college_ids = list(range(num_colleges))

    # Step 1: Initial college objects
    for c_id in college_ids:
        colleges[c_id] = {
            'capacity': capacity_per_college,
            'tentative_matches': [],
            'priority': lambda s: s  # simple identity priority
        }

    # Step 2: Each student ranks at least e * |B| colleges
    student_ranked = {}
    for s_id in student_ids:
        num_to_rank = max(1, int(e * num_colleges))
        ranked_colleges = random.sample(college_ids, num_to_rank)
        STUDENT_PREFERENCES[s_id] = {cid: rank for rank, cid in enumerate(ranked_colleges)}
        student_ranked[s_id] = set(ranked_colleges)

    # Step 3: Each college ranks at least e * |A| students
    college_ranked = {}
    for c_id in college_ids:
        num_to_rank = max(1, int(e * num_students))
        ranked_students = random.sample(student_ids, num_to_rank)
        COLLEGE_PREFERENCES[c_id] = {sid: rank for rank, sid in enumerate(ranked_students)}
        college_ranked[c_id] = set(ranked_students)

    # Step 4: Filter to only mutual rankings (edges exist only if both rank each other)
    for s_id in list(STUDENT_PREFERENCES.keys()):
        mutual_colleges = [cid for cid in STUDENT_PREFERENCES[s_id] if s_id in college_ranked.get(cid, set())]
        STUDENT_PREFERENCES[s_id] = {cid: rank for cid, rank in STUDENT_PREFERENCES[s_id].items() if cid in mutual_colleges}

    for c_id in list(COLLEGE_PREFERENCES.keys()):
        mutual_students = [sid for sid in COLLEGE_PREFERENCES[c_id] if c_id in student_ranked.get(sid, set())]
        COLLEGE_PREFERENCES[c_id] = {sid: rank for sid, rank in COLLEGE_PREFERENCES[c_id].items() if sid in mutual_students}

    # Step 5: Create student and college objects
    students = []
    for s_id in student_ids:
        students.append({
            'id': s_id,
            'preferences': list(STUDENT_PREFERENCES[s_id].keys()),
            'current_proposal_index': 0,
            'matched_college': None
        })

    return students, colleges
