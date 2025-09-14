import json
import os
import random

DATA_DIR = "data"

def make_square_json(n=10, out_dir=DATA_DIR):
    os.makedirs(out_dir, exist_ok=True)

    # Side A ranks side B
    A_to_B = {}
    for a in range(n):
        b_list = list(range(n))
        random.shuffle(b_list)
        A_to_B[a] = b_list

    # Side B ranks side A
    B_to_A = {}
    for b in range(n):
        a_list = list(range(n))
        random.shuffle(a_list)
        B_to_A[b] = a_list

    # Write JSON files
    with open(os.path.join(out_dir, "A_to_B.json"), "w") as f:
        json.dump(A_to_B, f, indent=2)
    with open(os.path.join(out_dir, "B_to_A.json"), "w") as f:
        json.dump(B_to_A, f, indent=2)

    print(f"Created {out_dir}/A_to_B.json and {out_dir}/B_to_A.json")

if __name__ == "__main__":
    make_square_json(10)
