"""
Training GCN and GCN+GAT models for balanced stable matching.
Based on WeaveNet paper experiments.
EXTENDED VERSION with 4 fairness loss variants including log-dif.
OPTIMIZED: 20 epochs, eval only at end
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data, Batch
import numpy as np
from typing import List, Tuple, Dict
import pickle
from tqdm import tqdm
import json
from pathlib import Path
import time

# Import the dataset generator
from dataset_generator import StableMatchingDataGenerator, MatchingInstance


class GCNModel(nn.Module):
    """Standard GCN for bipartite matching"""

    def __init__(self, num_layers: int = 6, hidden_dim: int = 64):
        super().__init__()
        self.num_layers = num_layers

        # Input: 2 features (satisfaction from A and B)
        self.input_proj = nn.Linear(2, hidden_dim)

        # GCN layers
        self.convs = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])

        # Output projection to edge weights (concatenated features: 2*hidden_dim)
        self.output_proj = nn.Linear(2 * hidden_dim, 1)

    def forward(self, x, edge_index, N, M):
        """
        Args:
            x: Node features [num_nodes, 2]
            edge_index: Edge connectivity [2, num_edges]
            N: Number of agents in group A
            M: Number of agents in group B
        Returns:
            matching: [N, M] matching matrix
        """
        # Initial projection
        x = F.relu(self.input_proj(x))

        # GCN layers with residual connections every 2 layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x_res = x
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)

            # Residual connection every 2 layers
            if i > 0 and i % 2 == 1:
                x = x + x_res

        # Compute edge features by concatenating node pairs
        edge_features = torch.cat([
            x[edge_index[0]],
            x[edge_index[1]]
        ], dim=1)

        # Project to matching scores
        edge_scores = self.output_proj(edge_features).squeeze(-1)

        # Reshape to matching matrix [N, M]
        matching = edge_scores.view(N, M)

        return matching


class GCNGATModel(nn.Module):
    """GCN layers followed by GAT layer"""

    def __init__(self, num_gcn_layers: int = 6, hidden_dim: int = 64,
                 gat_heads: int = 4):
        super().__init__()
        self.num_gcn_layers = num_gcn_layers

        # Input projection
        self.input_proj = nn.Linear(2, hidden_dim)

        # GCN layers
        self.gcn_convs = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(num_gcn_layers)
        ])

        self.gcn_batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_gcn_layers)
        ])

        # GAT layer
        self.gat = GATConv(hidden_dim, hidden_dim, heads=gat_heads, concat=False)
        self.gat_bn = nn.BatchNorm1d(hidden_dim)

        # Output projection (concatenated features: 2*hidden_dim)
        self.output_proj = nn.Linear(2 * hidden_dim, 1)

    def forward(self, x, edge_index, N, M):
        # Initial projection
        x = F.relu(self.input_proj(x))

        # GCN layers
        for i, (conv, bn) in enumerate(zip(self.gcn_convs, self.gcn_batch_norms)):
            x_res = x
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)

            if i > 0 and i % 2 == 1:
                x = x + x_res

        # GAT layer
        x = self.gat(x, edge_index)
        x = self.gat_bn(x)
        x = F.relu(x)

        # Compute edge features and matching
        edge_features = torch.cat([
            x[edge_index[0]],
            x[edge_index[1]]
        ], dim=1)

        edge_scores = self.output_proj(edge_features).squeeze(-1)
        matching = edge_scores.view(N, M)

        return matching


def instance_to_graph(instance: MatchingInstance) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert MatchingInstance to PyG graph format"""
    N, M = instance.N, instance.M

    # Create bipartite graph structure
    edge_index_list = []
    node_features = []

    # All edges connect A to B (complete bipartite graph)
    for i in range(N):
        for j in range(M):
            edge_index_list.append([i, N + j])

    # Node features: aggregate satisfaction scores
    for i in range(N):
        node_features.append([
            instance.satisfaction_A[i].mean(),
            0.0
        ])

    for j in range(M):
        node_features.append([
            0.0,
            instance.satisfaction_B[j].mean()
        ])

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t()
    x = torch.tensor(node_features, dtype=torch.float32)

    return x, edge_index


class FairnessLossVariant(nn.Module):
    """Four variants of fairness loss for stable matching"""

    def __init__(self, variant='squared', lambda_m=1.0, lambda_s=0.7, lambda_f=0.01):
        super().__init__()
        self.variant = variant  # 'squared', 'absolute', 'log_exp', 'log_dif'
        self.lambda_m = lambda_m
        self.lambda_s = lambda_s
        self.lambda_f = lambda_f

    def matrix_constraint_loss(self, m_hat):
        """Lm: Ensure matching is doubly stochastic"""
        m_A = F.softmax(m_hat, dim=1)
        m_B = F.softmax(m_hat, dim=0)

        def cosine_sim_matrix(m1, m2):
            m1_norm = F.normalize(m1, p=2, dim=1)
            m2_norm = F.normalize(m2.t(), p=2, dim=1)
            cos_sim = (m1_norm * m2_norm).sum(dim=1).mean()
            return cos_sim

        c_AB = cosine_sim_matrix(m_A, m_B)
        c_BA = cosine_sim_matrix(m_B.t(), m_A.t())

        loss = 1 - (c_AB + c_BA) / 2
        return loss

    def blocking_pair_loss(self, m_hat, sat_A, sat_B):
        """Ls: Penalize blocking pairs"""
        m_A = F.softmax(m_hat, dim=1)
        N, M = m_hat.shape

        total_loss = 0.0
        for i in range(N):
            for j in range(M):
                g_A = 0.0
                for k in range(M):
                    if k != j:
                        envy = sat_A[i, j] - sat_A[i, k]
                        if envy > 0:
                            g_A += m_A[i, k] * envy

                g_B = 0.0
                for k in range(N):
                    if k != i:
                        envy = sat_B[j, i] - sat_B[j, k]
                        if envy > 0:
                            g_B += m_A[k, j] * envy

                total_loss += g_A * g_B

        return total_loss / (N * M)

    def fairness_loss(self, m_hat, sat_A, sat_B):
        """Lf: Four variants of fairness measurement"""
        m_A = F.softmax(m_hat, dim=1)
        N = m_hat.shape[0]

        S_A = (m_A * sat_A).sum()
        S_B = (m_A * sat_B.t()).sum()
        diff = torch.abs(S_A - S_B)

        if self.variant == 'squared':
            # L_f^sq = (1/N^2) * (S_A - S_B)^2
            return ((S_A - S_B) ** 2) / (N ** 2)

        elif self.variant == 'absolute':
            # L_f^abs = (1/N) * |S_A - S_B|
            return diff / N

        elif self.variant == 'log_exp':
            # L_f^exp = log(1 + exp(|S_A - S_B| / N))
            return torch.log(1 + torch.exp(diff / N))

        elif self.variant == 'log_dif':
            # L_f^log-dif = ln(S_A) + ln(S_B) + |S_A - S_B|
            # Add small epsilon to avoid log(0)
            eps = 1e-8
            return torch.log(S_A + eps) + torch.log(S_B + eps) + diff

        else:
            raise ValueError(f"Unknown variant: {self.variant}")

    def forward(self, m_hat, sat_A, sat_B):
        """Combined loss (Eq. 6)"""
        L_m = self.matrix_constraint_loss(m_hat)

        L_s_A = self.blocking_pair_loss(m_hat, sat_A, sat_B)
        L_f_A = self.fairness_loss(m_hat, sat_A, sat_B)

        L_s_B = self.blocking_pair_loss(m_hat.t(), sat_B, sat_A)
        L_f_B = self.fairness_loss(m_hat.t(), sat_B, sat_A)

        total_loss = (self.lambda_m * L_m +
                     0.5 * (self.lambda_s * (L_s_A + L_s_B) +
                            self.lambda_f * (L_f_A + L_f_B)))

        return total_loss, {
            'L_m': L_m.item(),
            'L_s': ((L_s_A + L_s_B) / 2).item(),
            'L_f': ((L_f_A + L_f_B) / 2).item()
        }


def train_epoch(model, generator, loss_fn, optimizer, N_train, num_samples,
                distributions, device):
    """Train for one epoch with on-the-fly data generation"""
    model.train()
    total_loss = 0
    loss_components = {'L_m': 0, 'L_s': 0, 'L_f': 0}

    for _ in range(num_samples):
        dist = np.random.choice(distributions)
        instance = generator.generate_instance(N_train, N_train, dist)

        x, edge_index = instance_to_graph(instance)
        x = x.to(device)
        edge_index = edge_index.to(device)

        sat_A = torch.tensor(instance.satisfaction_A, dtype=torch.float32, device=device)
        sat_B = torch.tensor(instance.satisfaction_B, dtype=torch.float32, device=device)

        m_hat = model(x, edge_index, N_train, N_train)
        loss, components = loss_fn(m_hat, sat_A, sat_B)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        for key in loss_components:
            loss_components[key] += components[key]

    avg_loss = total_loss / num_samples
    for key in loss_components:
        loss_components[key] /= num_samples

    return avg_loss, loss_components


def compute_weavenet_metrics(m_binary, instance):
    """Compute SEq, Reg, Egal, Bal metrics from WeaveNet paper"""
    m_np = m_binary.cpu().numpy()
    N, M = instance.N, instance.M

    # P(m; A) and P(m; B)
    P_A = sum(instance.preference_A[i, j] for i in range(N)
             for j in range(M) if m_np[i, j] == 1)
    P_B = sum(instance.preference_B[j, i] for i in range(N)
             for j in range(M) if m_np[i, j] == 1)

    SEq = abs(P_A - P_B)
    Egal = P_A + P_B
    Bal = max(P_A, P_B)

    # Regret: worst rank in the matching
    Reg = 0
    for i in range(N):
        for j in range(M):
            if m_np[i, j] == 1:
                Reg = max(Reg, max(instance.preference_A[i, j], instance.preference_B[j, i]))

    return {'SEq': SEq, 'Reg': Reg, 'Egal': Egal, 'Bal': Bal}


def compute_efk(m_binary, instance):
    """Compute EF-k: max envy violations"""
    m_np = m_binary.cpu().numpy()
    N = instance.N

    max_envy = 0
    for s in range(N):
        s_match_idx = np.argmax(m_np[s])
        s_rank = instance.preference_A[s, s_match_idx]

        envy_count = 0
        for s_prime in range(N):
            if s_prime == s:
                continue
            sp_match_idx = np.argmax(m_np[s_prime])
            sp_rank_by_s = instance.preference_A[s, sp_match_idx]

            # Does s prefer s_prime's match?
            if sp_rank_by_s < s_rank:
                # Does the university prefer s_prime over s?
                if instance.preference_B[sp_match_idx, s_prime] < instance.preference_B[sp_match_idx, s]:
                    envy_count += 1

        max_envy = max(max_envy, envy_count)

    return max_envy


def compute_ard_esd_wlr(m_binary, instance):
    """Compute ARD, ESD, WLR metrics"""
    m_np = m_binary.cpu().numpy()
    N, M = instance.N, instance.M

    rank_diffs = []
    esds = []
    wlrs = []

    for i in range(N):
        j = np.argmax(m_np[i])
        M_u = instance.preference_A[i, j]
        M_v = instance.preference_B[j, i]

        rank_diffs.append(abs(M_u - M_v))
        esds.append(np.exp(abs(M_u - M_v)))
        wlrs.append(M_u * np.log(M_u + 1) + M_v * np.log(M_v + 1) + abs(M_u - M_v))

    ARD = np.mean(rank_diffs) if rank_diffs else 0
    ESD = np.max(esds) if esds else 0
    WLR = np.max(wlrs) if wlrs else 0

    return {'ARD': ARD, 'ESD': ESD, 'WLR': WLR}


def evaluate_model(model, test_instances, device, debug=False):
    """Evaluate on test set with comprehensive metrics including worst-case"""
    model.eval()

    # Aggregate metrics
    agg_results = {
        'stable_rate': 0,
        'avg_blocking_pairs': 0,
        'avg_EF_k': 0,
        'avg_SEq': 0,
        'avg_Reg': 0,
        'avg_Egal': 0,
        'avg_Bal': 0,
        'avg_ARD': 0,
        'avg_ESD': 0,
        'avg_WLR': 0,
    }

    # Worst-case metrics (track maximum across all instances)
    worst_results = {
        'worst_blocking_pairs': 0,
        'worst_EF_k': 0,
        'worst_SEq': 0,
        'worst_Reg': 0,
        'worst_Egal': 0,
        'worst_Bal': 0,
        'worst_ARD': 0,
        'worst_ESD': 0,
        'worst_WLR': 0,
    }

    with torch.no_grad():
        for idx, instance in enumerate(test_instances):
            x, edge_index = instance_to_graph(instance)
            x = x.to(device)
            edge_index = edge_index.to(device)

            m_hat = model(x, edge_index, instance.N, instance.M)

            # Binarize with argmax per row
            m_binary = torch.zeros_like(m_hat)
            row_max = m_hat.argmax(dim=1)
            m_binary[torch.arange(instance.N), row_max] = 1
            m_np = m_binary.cpu().numpy()

            # Count blocking pairs
            num_blocking = 0
            for i in range(instance.N):
                j_curr = np.argmax(m_np[i])

                for j in range(instance.M):
                    if j == j_curr:
                        continue

                    if instance.preference_A[i, j] >= instance.preference_A[i, j_curr]:
                        continue

                    students_at_j = np.where(m_np[:, j] == 1)[0]

                    for i_curr in students_at_j:
                        if instance.preference_B[j, i] < instance.preference_B[j, i_curr]:
                            num_blocking += 1
                            break

            is_stable = (num_blocking == 0)
            agg_results['stable_rate'] += is_stable
            agg_results['avg_blocking_pairs'] += num_blocking
            worst_results['worst_blocking_pairs'] = max(worst_results['worst_blocking_pairs'], num_blocking)

            # WeaveNet metrics
            wn_metrics = compute_weavenet_metrics(m_binary, instance)
            agg_results['avg_SEq'] += wn_metrics['SEq']
            agg_results['avg_Reg'] += wn_metrics['Reg']
            agg_results['avg_Egal'] += wn_metrics['Egal']
            agg_results['avg_Bal'] += wn_metrics['Bal']

            worst_results['worst_SEq'] = max(worst_results['worst_SEq'], wn_metrics['SEq'])
            worst_results['worst_Reg'] = max(worst_results['worst_Reg'], wn_metrics['Reg'])
            worst_results['worst_Egal'] = max(worst_results['worst_Egal'], wn_metrics['Egal'])
            worst_results['worst_Bal'] = max(worst_results['worst_Bal'], wn_metrics['Bal'])

            # EF-k
            efk = compute_efk(m_binary, instance)
            agg_results['avg_EF_k'] += efk
            worst_results['worst_EF_k'] = max(worst_results['worst_EF_k'], efk)

            # ARD, ESD, WLR
            pair_metrics = compute_ard_esd_wlr(m_binary, instance)
            agg_results['avg_ARD'] += pair_metrics['ARD']
            agg_results['avg_ESD'] += pair_metrics['ESD']
            agg_results['avg_WLR'] += pair_metrics['WLR']

            worst_results['worst_ARD'] = max(worst_results['worst_ARD'], pair_metrics['ARD'])
            worst_results['worst_ESD'] = max(worst_results['worst_ESD'], pair_metrics['ESD'])
            worst_results['worst_WLR'] = max(worst_results['worst_WLR'], pair_metrics['WLR'])

    num_instances = len(test_instances)
    for key in agg_results:
        agg_results[key] /= num_instances

    # Convert all numpy types to Python native types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, (np.int64, np.int32, np.int_)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float_)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj

    agg_results = convert_to_native(agg_results)
    worst_results = convert_to_native(worst_results)

    # Combine average and worst-case metrics
    return {**agg_results, **worst_results}


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Training parameters
    N_train = 10
    test_sizes = [3, 5, 7, 9]
    distributions = ["UU", "DD", "GG", "UD", "Lib"]

    num_epochs = 20  # REDUCED TO 20
    samples_per_epoch = 1000

    # Loss variants to test (including log_dif)
    loss_variants = ['squared', 'absolute', 'log_exp', 'log_dif']

    # Base model configurations
    base_configs = [
        {'name': 'GCN-6', 'type': 'gcn', 'layers': 6},
        {'name': 'GCN-16', 'type': 'gcn', 'layers': 16},
        {'name': 'GCN-32', 'type': 'gcn', 'layers': 32},
        {'name': 'GCN6-GAT1', 'type': 'gat', 'layers': 6},
        {'name': 'GCN16-GAT1', 'type': 'gat', 'layers': 16},
        {'name': 'GCN32-GAT1', 'type': 'gat', 'layers': 32},
    ]

    # Create full configs: each base config × each loss variant
    configs = []
    for base_config in base_configs:
        for loss_var in loss_variants:
            configs.append({
                **base_config,
                'loss_variant': loss_var,
                'full_name': f"{base_config['name']}-{loss_var}"
            })

    print(f"\n{'='*60}")
    print(f"Total configurations to train: {len(configs)}")
    print(f"Epochs per config: {num_epochs}")
    print(f"Estimated total time: ~{len(configs) * 0.2:.1f} hours")
    print(f"{'='*60}\n")

    generator = StableMatchingDataGenerator(seed=42)

    # Generate test data
    print("\nGenerating test data...")
    test_data = {}
    test_data_by_dist = {}  # Store by distribution for analysis

    for N in test_sizes:
        test_instances = []
        test_by_dist = {dist: [] for dist in distributions}

        for dist in distributions:
            for _ in range(200):
                instance = generator.generate_instance(N, N, dist)
                test_instances.append(instance)
                test_by_dist[dist].append(instance)

        test_data[N] = test_instances
        test_data_by_dist[N] = test_by_dist
        print(f"  Generated {len(test_instances)} test instances for N={N}")

    # Train each configuration
    all_results = {}

    for config_idx, config in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Training {config['full_name']} [{config_idx+1}/{len(configs)}]")
        print(f"Architecture: {config['type'].upper()}, Layers: {config['layers']}, Loss: {config['loss_variant']}")
        print(f"{'='*60}")

        if config['type'] == 'gcn':
            model = GCNModel(num_layers=config['layers']).to(device)
        else:
            model = GCNGATModel(num_gcn_layers=config['layers']).to(device)

        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {num_params:,}")

        loss_fn = FairnessLossVariant(variant=config['loss_variant'])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        start_time = time.time()

        # Training loop - NO EVALUATION DURING TRAINING
        for epoch in range(num_epochs):
            avg_loss, components = train_epoch(
                model, generator, loss_fn, optimizer, N_train,
                samples_per_epoch, distributions, device
            )

            # Print progress every 5 epochs
            if (epoch + 1) % 5 == 0:
                elapsed = time.time() - start_time
                print(f"  Epoch {epoch+1}/{num_epochs} ({elapsed/60:.1f}min) - "
                      f"Loss: {avg_loss:.4f} [L_m: {components['L_m']:.4f}, "
                      f"L_s: {components['L_s']:.4f}, L_f: {components['L_f']:.4f}]")

        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time/60:.1f} minutes")

        # EVALUATE ONLY AT THE END
        print(f"\nFinal Evaluation for {config['full_name']}:")
        config_results = {}

        eval_start = time.time()

        # Evaluate on all test sizes
        for N in test_sizes:
            results = evaluate_model(model, test_data[N], device)
            config_results[f'N={N}'] = {
                'overall': results
            }

            # Also evaluate per distribution
            dist_results = {}
            for dist in distributions:
                dist_res = evaluate_model(model, test_data_by_dist[N][dist], device)
                dist_results[dist] = dist_res

            config_results[f'N={N}']['by_distribution'] = dist_results

            print(f"  N={N}: Stable={results['stable_rate']:.2%}, "
                  f"EF-k(avg/worst)={results['avg_EF_k']:.2f}/{results['worst_EF_k']:.0f}, "
                  f"SEq(avg/worst)={results['avg_SEq']:.2f}/{results['worst_SEq']:.2f}")

        eval_time = time.time() - eval_start
        print(f"Evaluation completed in {eval_time/60:.1f} minutes")

        all_results[config['full_name']] = config_results

        # Save model
        torch.save(model.state_dict(), f"models/{config['full_name']}_final.pt")

        # Save intermediate results after each config
        with open("results/gcn_all_losses_partial.json", 'w') as f:
            json.dump(all_results, f, indent=2)

    # Save final results
    Path("results").mkdir(exist_ok=True)
    with open("results/gcn_all_losses.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*60)
    print("Training complete! Results saved to results/gcn_all_losses.json")
    print("="*60)


if __name__ == "__main__":
    Path("models").mkdir(exist_ok=True)
    main()
