import json
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from weavenet.model import TrainableMatchingModule, WeaveNet
from weavenet.metric import binarize

# --- Monkey-patch: neutralize crashing stability loss inside criteria ---
import weavenet.criteria as _crit_mod
def _noop_stability(m, sab, sba):
    B = m.shape[0]
    return torch.zeros(B, device=m.device, dtype=m.dtype)
_crit_mod.loss_stability = _noop_stability

from weavenet.criteria import CriteriaStableMatching  # import after patch

# -------------------------
# Utils
# -------------------------
def _is_perm(lst, upto):
    return isinstance(lst, list) and len(lst) == upto and sorted(lst) == list(range(upto))

def set_seed(seed=1337):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# IO + preprocessing
# -------------------------
def load_json_rankings(json_file_a_to_b, json_file_b_to_a):
    with open(json_file_a_to_b, "r") as f:
        data_a_to_b = json.load(f)
    with open(json_file_b_to_a, "r") as f:
        data_b_to_a = json.load(f)

    keys_a = sorted(data_a_to_b.keys(), key=lambda k: int(k))
    keys_b = sorted(data_b_to_a.keys(), key=lambda k: int(k))

    nA = len(keys_a); nB = len(keys_b)
    assert nA > 0 and nB > 0, "Empty rankings."

    sample_a = data_a_to_b[keys_a[0]]
    sample_b = data_b_to_a[keys_b[0]]
    mA = len(sample_a); mB = len(sample_b)
    assert mA == nB and mB == nA, (
        f"A→B lists length {mA} must equal |B|={nB}; "
        f"B→A lists length {mB} must equal |A|={nA}."
    )

    for a in keys_a:
        if not _is_perm(data_a_to_b[a], nB):
            raise ValueError(f"A[{a}] list must be a permutation of 0..{nB-1}. Got {data_a_to_b[a]}")
    for b in keys_b:
        if not _is_perm(data_b_to_a[b], nA):
            raise ValueError(f"B[{b}] list must be a permutation of 0..{nA-1}. Got {data_b_to_a[b]}")

    # ranks: lower is better (0 best)
    ranks_AtoB = np.zeros((nA, nB), dtype=np.float32)
    for a in keys_a:
        a_id = int(a)
        for r, b_id in enumerate(data_a_to_b[a]):
            ranks_AtoB[a_id, b_id] = float(r)

    ranks_BtoA = np.zeros((nB, nA), dtype=np.float32)
    for b in keys_b:
        b_id = int(b)
        for r, a_id in enumerate(data_b_to_a[b]):
            ranks_BtoA[b_id, a_id] = float(r)

    print(f"Loaded: {nA} A-agents × {nB} B-agents.")
    return ranks_AtoB, ranks_BtoA  # [N,M] and [M,N]

def convert_to_satisfaction(ranks_AtoB, ranks_BtoA):
    """
    Build six tensors:
      sab4   [B,N,M,1]  A→B for model & loss
      sab3   [B,N,M]    A→B for metrics
      sba4   [B,M,N,1]  B→A (UNTRANSPOSED) for loss
      sba3   [B,M,N]    B→A (UNTRANSPOSED)
      sba_t4 [B,N,M,1]  (B→A)^T for model forward
      sba_t3 [B,N,M]    (B→A)^T for metrics
    """
    N, M = ranks_AtoB.shape
    M2, N2 = ranks_BtoA.shape
    assert (N, M) == (N2, M2)[::-1], f"Shapes disagree: A→B={ranks_AtoB.shape}, B→A={ranks_BtoA.shape}"

    max_rank_A = max(M - 1, 1)
    max_rank_B = max(N - 1, 1)
    sab_np = 1.0 - (ranks_AtoB / max_rank_A)  # [N,M]
    sba_np = 1.0 - (ranks_BtoA / max_rank_B)  # [M,N]

    sab4   = torch.tensor(sab_np, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)     # [1,N,M,1]
    sab3   = sab4.squeeze(-1)                                                         # [1,N,M]
    sba4   = torch.tensor(sba_np, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)     # [1,M,N,1]
    sba3   = sba4.squeeze(-1)                                                         # [1,M,N]
    sba_t4 = torch.tensor(sba_np.T, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)   # [1,N,M,1]
    sba_t3 = sba_t4.squeeze(-1)                                                       # [1,N,M]

    return (
        sab4.contiguous(), sab3.contiguous(),
        sba4.contiguous(), sba3.contiguous(),
        sba_t4.contiguous(), sba_t3.contiguous()
    )

# -------------------------
# Model
# -------------------------
def create_weavenet_model(input_channels=2):
    return TrainableMatchingModule(
        net=WeaveNet(
            input_channels=input_channels,
            output_channels_list=[32, 32, 32, 32, 32, 32],
            mid_channels_list=[64, 64, 64, 64, 64, 64],
            calc_residual=[False] * 6,
            keep_first_var_after=0,
        ),
        output_channels=1,
    )

# -------------------------
# Training
# -------------------------
def train(model, sab4, sab3, sba4, sba3, sba_t4, sba_t3, epochs=100, lr=1e-3, grad_clip=1.0, device=None):
    """
    Forward: (sab4, sba_t4)      -> both [B,N,M,1]
    Loss:    (sigmoid(logits), sab4, sba4)  -> pass PROBABILITIES
    Metrics: (sigmoid(logits), sab3, sba_t3)
    """
    device = device or _device()
    model.to(device)
    sab4, sab3 = sab4.to(device), sab3.to(device)
    sba4, sba3 = sba4.to(device), sba3.to(device)
    sba_t4, sba_t3 = sba_t4.to(device), sba_t3.to(device)

    criteria = CriteriaStableMatching(
        one2one_weight=5.0,      # emphasize 1-1
        stability_weight=0.7,    # patched to no-op
        fairness="sexequality",
        fairness_weight=0.0,     # focus on 1-1 first; re-enable later
    )
    loss_fn = criteria.generate_criterion()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    print("Starting training...")
    for e in range(epochs):
        model.train()
        opt.zero_grad()

        try:
            out = model(sab4, sba_t4)
        except TypeError:
            inputs = torch.cat([sab4, sba_t4], dim=-1)  # [B,N,M,2]
            out = model(inputs)

        logits4d = out[0] if isinstance(out, (tuple, list)) else out  # [B,N,M,1] or [B,N,M]
        if logits4d.dim() == 3:
            logits4d = logits4d.unsqueeze(-1)

        # --- convert to probabilities for loss/metrics ---
        probs4d = torch.sigmoid(logits4d)          # [B,N,M,1]
        probs2d = probs4d.squeeze(-1)              # [B,N,M]

        # Loss — use probabilities (repo losses expect [0,1] mass)
        loss, log = loss_fn(probs4d.contiguous(), sab4.contiguous(), sba4.contiguous())
        loss.mean().backward()
        clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        opt.step()

        # Metrics — use probabilities
        if e % 20 == 0:
            with torch.no_grad():
                metrics, _ = criteria.metric(probs2d, sab3, sba_t3)  # use sba_t3 (transposed)
                print(
                    f"Epoch {e:3d} | "
                    f"loss {loss.mean().item():.4f} | "
                    f"1-1 {metrics['is_one2one'].float().mean():.3f} | "
                    f"stable {metrics['is_stable'].float().mean():.3f} | "
                    f"success {metrics['is_success'].float().mean():.3f}"
                )
    return model

# -------------------------
# Evaluation
# -------------------------
def final_matching_and_metrics(model, sab4, sab3, sba4, sba3, sba_t4, sba_t3):
    device = next(model.parameters()).device
    sab4, sab3 = sab4.to(device), sab3.to(device)
    sba4, sba3 = sba4.to(device), sba3.to(device)
    sba_t4, sba_t3 = sba_t4.to(device), sba_t3.to(device)

    criteria = CriteriaStableMatching(
        one2one_weight=5.0,
        stability_weight=0.7,
        fairness="sexequality",
        fairness_weight=0.0,
    )

    with torch.no_grad():
        try:
            out = model(sab4, sba_t4)
        except TypeError:
            inputs = torch.cat([sab4, sba_t4], dim=-1)
            out = model(inputs)

        logits4d = out[0] if isinstance(out, (tuple, list)) else out
        if logits4d.dim() == 3:
            logits4d = logits4d.unsqueeze(-1)

        probs4d = torch.sigmoid(logits4d)
        probs2d = probs4d.squeeze(-1)  # [B,N,M]

        # binarize from probabilities
        try:
            bm = binarize(probs2d)
        except Exception:
            bm = binarize(torch.sigmoid(probs2d))

        metrics, _ = criteria.metric(probs2d, sab3, sba_t3)

    return probs2d.cpu(), bm.cpu(), metrics

# -------------------------
# Pipeline
# -------------------------
def run(json_file_a_to_b, json_file_b_to_a, epochs=100, seed=1337):
    print("=" * 60)
    print("WEAVENET BIPARTITE MATCHING")
    print("=" * 60)

    set_seed(seed)
    ranks_AtoB, ranks_BtoA = load_json_rankings(json_file_a_to_b, json_file_b_to_a)

    print("\nConverting ranks → satisfaction...")
    sab4, sab3, sba4, sba3, sba_t4, sba_t3 = convert_to_satisfaction(ranks_AtoB, ranks_BtoA)
    N, M = sab3.shape[1:]

    print("\nCreating model...")
    model = create_weavenet_model(input_channels=2)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total:,}")

    print(f"\nTraining for {epochs} epochs...")
    model = train(model, sab4, sab3, sba4, sba3, sba_t4, sba_t3, epochs=epochs)

    print("\nEvaluating final matching...")
    probs, bm, metrics = final_matching_and_metrics(model, sab4, sab3, sba4, sba3, sba_t4, sba_t3)

    print("\nFINAL RESULTS")
    print(f"One-to-one: {metrics['is_one2one'].float().mean().item():.3f}")
    print(f"Stability:  {metrics['is_stable'].float().mean().item():.3f}")   # metric only (loss stability is noop)
    print(f"Success:    {metrics['is_success'].float().mean().item():.3f}")

    match = bm[0].int().numpy()  # [N,M]
    print("\nBinary matching matrix (1 = matched):")
    header = "    " + " ".join([f"B{j}" for j in range(M)])
    print(header)
    for i in range(N):
        row = " ".join(str(match[i, j]) for j in range(M))
        print(f"A{i}: {row}")

    pairs = np.argwhere(match == 1)
    print(f"\nMatching pairs ({len(pairs)} total):")
    for i, j in pairs:
        sab_score = sab4[0, i, j, 0].item()
        sba_score = sba4[0, j, i, 0].item()
        print(f"  A{i} ↔ B{j} (satisfaction A→B={sab_score:.3f}, B→A={sba_score:.3f})")

    print("\n" + "=" * 60)
    return model, bm

if __name__ == "__main__":
    file_a_to_b = "/home/beloslava/Desktop/tue/honors/FairAndBalancedBipartiteGraphs/paper/data/A_to_B.json"
    file_b_to_a = "/home99999999999999999999999999999999999999999999999999999999999999999999990/beloslava/Desktop/tue/honors/FairAndBalancedBipartiteGraphs/paper/data/B_to_A.json"
    run(file_a_to_b, file_b_to_a, epochs=100)
