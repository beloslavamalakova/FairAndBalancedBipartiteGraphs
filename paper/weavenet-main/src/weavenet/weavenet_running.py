import torch
import numpy as np
import json
from model import TrainableMatchingModule, WeaveNet
from preference import PreferenceFormat, to_rank
from criteria import CriteriaStableMatching
from metric import binarize

def load_json_rankings(json_file_a_to_b, json_file_b_to_a):
    """
    Load ranking data from JSON files.

    Args:
        json_file_a_to_b: Path to JSON file with rankings from side A to B
        json_file_b_to_a: Path to JSON file with rankings from side B to A

    Returns:
        rankings_a_to_b: Numpy array [N, M] where N agents rank M targets
        rankings_b_to_a: Numpy array [M, N] where M agents rank N targets
    """
    with open(json_file_a_to_b, 'r') as f:
        data_a_to_b = json.load(f)

    with open(json_file_b_to_a, 'r') as f:
        data_b_to_a = json.load(f)

    # Convert JSON dict to numpy arrays
    n_agents_a = len(data_a_to_b)
    n_agents_b = len(data_b_to_a)

    print(f"Found {n_agents_a} agents in group A, {n_agents_b} agents in group B")

    # Get the number of targets each agent ranks
    sample_a_prefs = list(data_a_to_b.values())[0]
    sample_b_prefs = list(data_b_to_a.values())[0]
    n_targets_by_a = len(sample_a_prefs)  # Number of B agents that A agents rank
    n_targets_by_b = len(sample_b_prefs)  # Number of A agents that B agents rank

    print(f"Group A agents rank {n_targets_by_a} targets, Group B agents rank {n_targets_by_b} targets")

    # For your case: both should be 10
    assert n_agents_a == 10 and n_agents_b == 10, f"Expected 10 agents in each group, got A:{n_agents_a}, B:{n_agents_b}"
    assert n_targets_by_a == 10 and n_targets_by_b == 10, f"Expected each agent to rank 10 targets, got A→B:{n_targets_by_a}, B→A:{n_targets_by_b}"

    # Initialize ranking matrices
    rankings_a_to_b = np.zeros((n_agents_a, n_targets_by_a))  # [10, 10]
    rankings_b_to_a = np.zeros((n_agents_b, n_targets_by_b))  # [10, 10]

    # Convert preference lists to rank positions for A→B
    for agent_id_str in data_a_to_b:
        agent_id = int(agent_id_str)
        pref_list = data_a_to_b[agent_id_str]

        # Convert preference list to rank positions
        # pref_list = [7, 2, 6, 9, 3, 8, 5, 4, 1, 0] means:
        # target 7 is ranked 0 (most preferred), target 2 is ranked 1, etc.
        for rank, target_id in enumerate(pref_list):
            rankings_a_to_b[agent_id, target_id] = rank

    # Convert preference lists to rank positions for B→A
    for agent_id_str in data_b_to_a:
        agent_id = int(agent_id_str)
        pref_list = data_b_to_a[agent_id_str]

        for rank, target_id in enumerate(pref_list):
            rankings_b_to_a[agent_id, target_id] = rank

    print(f"Rankings A→B shape: {rankings_a_to_b.shape}")
    print(f"Rankings B→A shape: {rankings_b_to_a.shape}")

    # Verify the rankings make sense
    print(f"A→B rankings range: {rankings_a_to_b.min():.0f} to {rankings_a_to_b.max():.0f}")
    print(f"B→A rankings range: {rankings_b_to_a.min():.0f} to {rankings_b_to_a.max():.0f}")

    return rankings_a_to_b, rankings_b_to_a

def convert_rankings_to_satisfaction(rankings_a_to_b, rankings_b_to_a):
    """
    Convert ranking matrices to satisfaction matrices for WeaveNet.

    Args:
        rankings_a_to_b: Rankings from side A to B [N, M] - lower rank = more preferred
        rankings_b_to_a: Rankings from side B to A [M, N] - lower rank = more preferred

    Returns:
        sab: Satisfaction matrix A to B [1, N, M, 1] (with batch dimension)
        sba_t: Satisfaction matrix B to A transposed [1, N, M, 1]
    """
    N, M = rankings_a_to_b.shape  # Should be [10, 10]
    M_check, N_check = rankings_b_to_a.shape  # Should be [10, 10]

    print(f"Converting rankings: A→B {rankings_a_to_b.shape}, B→A {rankings_b_to_a.shape}")

    # Verify dimensions match
    assert N == N_check and M == M_check, f"Dimension mismatch: A→B is [{N},{M}], B→A is [{M_check},{N_check}]"

    # Convert rankings to satisfaction scores
    # Ranking 0 = most preferred → satisfaction = 1.0
    # Ranking 9 = least preferred → satisfaction = 0.0
    max_rank = 9.0  # Rankings go from 0 to 9

    sab = 1.0 - (rankings_a_to_b / max_rank)  # [10, 10]
    sba = 1.0 - (rankings_b_to_a / max_rank)  # [10, 10]

    print(f"Satisfaction ranges: A→B [{sab.min():.2f}, {sab.max():.2f}], B→A [{sba.min():.2f}, {sba.max():.2f}]")

    # Convert to PyTorch tensors and add batch + channel dimensions
    sab = torch.tensor(sab, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # [1, 10, 10, 1]

    # Transpose B→A from [10, 10] to [10, 10] and add dimensions
    # Note: sba is already [M,N] = [10,10] where M=10 B-agents rank N=10 A-agents
    # We need sba_t to be [N,M] = [10,10] where we transpose to get A-agents × B-agents format
    sba_t = torch.tensor(sba.T, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # [1, 10, 10, 1]

    print(f"Final tensor shapes: sab {sab.shape}, sba_t {sba_t.shape}")

    return sab, sba_t

def create_weavenet_model(input_channels=2):
    """Create a WeaveNet model for stable matching."""
    model = TrainableMatchingModule(
        net=WeaveNet(
            input_channels=input_channels,
            output_channels_list=[32, 32, 32, 32, 32, 32],
            mid_channels_list=[64, 64, 64, 64, 64, 64],
            calc_residual=[False] * 6,
            keep_first_var_after=0,
        ),
        output_channels=1,
    )
    return model

def train_on_your_data(model, sab, sba_t, num_epochs=100):
    """
    Training loop for bipartite matching data.
    """
    # Set up loss criteria
    criteria = CriteriaStableMatching(
        one2one_weight=1.0,
        stability_weight=0.7,
        fairness='sexequality',
        fairness_weight=0.1
    )
    criterion_fn = criteria.generate_criterion()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Starting training...")
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Forward pass
        matching_logits = model(sab, sba_t)  # Returns [1, 10, 10, 1]

        # Calculate loss
        loss, log = criterion_fn(matching_logits, sab.squeeze(-1), sba_t.squeeze(-1))

        # Backward pass
        loss.mean().backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}: Loss = {loss.mean().item():.4f}")

            # Calculate metrics
            with torch.no_grad():
                metrics, binary_matching = criteria.metric(
                    matching_logits.squeeze(-1),
                    sab.squeeze(-1),
                    sba_t.squeeze(-1)
                )
                print(f"           One-to-one: {metrics['is_one2one'].float().mean():.3f}")
                print(f"           Stable: {metrics['is_stable'].float().mean():.3f}")
                print(f"           Success: {metrics['is_success'].float().mean():.3f}")

    return model

def run_bipartite_matching(json_file_a_to_b, json_file_b_to_a, num_epochs=100):
    """
    Complete pipeline to load JSON data and train WeaveNet for bipartite matching.

    Args:
        json_file_a_to_b: JSON file with rankings from group A to group B
        json_file_b_to_a: JSON file with rankings from group B to group A
        num_epochs: Number of training epochs
    """
    print("="*60)
    print("WEAVENET BIPARTITE MATCHING")
    print("="*60)

    # Load data
    print("\n1. Loading JSON ranking data...")
    rankings_a_to_b, rankings_b_to_a = load_json_rankings(json_file_a_to_b, json_file_b_to_a)

    # Convert to satisfaction matrices
    print("\n2. Converting to satisfaction matrices...")
    sab, sba_t = convert_rankings_to_satisfaction(rankings_a_to_b, rankings_b_to_a)

    # Create model
    print("\n3. Creating WeaveNet model...")
    model = create_weavenet_model(input_channels=2)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {total_params:,} parameters")

    # Train model
    print(f"\n4. Training model for {num_epochs} epochs...")
    trained_model = train_on_your_data(model, sab, sba_t, num_epochs=num_epochs)

    # Get final matching
    print("\n5. Generating final matching...")
    with torch.no_grad():
        matching_output = trained_model(sab, sba_t)
        matching_logits = matching_output.squeeze(-1)  # Remove channel dimension [1, 10, 10]
        binary_matching = binarize(matching_logits)

        print(f"Final matching shape: {binary_matching.shape}")

        # Calculate final metrics
        criteria = CriteriaStableMatching(
            one2one_weight=1.0,
            stability_weight=0.7,
            fairness='sexequality',
            fairness_weight=0.1
        )
        final_metrics, _ = criteria.metric(
            matching_logits,
            sab.squeeze(-1),
            sba_t.squeeze(-1)
        )

        print("\nFINAL RESULTS:")
        print(f"One-to-one constraint: {final_metrics['is_one2one'].float().mean():.3f}")
        print(f"Stability: {final_metrics['is_stable'].float().mean():.3f}")
        print(f"Overall success: {final_metrics['is_success'].float().mean():.3f}")

        # Show the matching
        print(f"\nBinary matching matrix (1 = matched, 0 = not matched):")
        matching_matrix = binary_matching[0].int().numpy()
        print("   ", " ".join([f"B{j}" for j in range(10)]))
        for i in range(10):
            print(f"A{i}: {' '.join([str(matching_matrix[i,j]) for j in range(10)])}")

        # Show matching pairs
        matches = torch.where(binary_matching[0] == 1)
        print(f"\nMatching pairs ({len(matches[0])} total):")
        if len(matches[0]) > 0:
            for i, j in zip(matches[0].numpy(), matches[1].numpy()):
                # Show the satisfaction scores for this match
                sab_score = sab[0, i, j, 0].item()
                sba_score = sba_t[0, i, j, 0].item()
                print(f"  A{i} ↔ B{j} (satisfaction: A→B={sab_score:.3f}, B→A={sba_score:.3f})")
        else:
            print("  No matches found!")

    print("\n" + "="*60)
    return trained_model, binary_matching

# Usage for your specific case:
if __name__ == "__main__":
    # Replace these paths with your actual file paths
    file_a_to_b = "/home/beloslava/Desktop/tue/honors/FairAndBalancedBipartiteGraphs/paper/data/group_a_to_b.json"
    file_b_to_a = "/home/beloslava/Desktop/tue/honors/FairAndBalancedBipartiteGraphs/paper/data/group_b_to_a.json"

    # Run the complete pipeline
    trained_model, final_matching = run_bipartite_matching(
        file_a_to_b,
        file_b_to_a,
        num_epochs=100
    )

