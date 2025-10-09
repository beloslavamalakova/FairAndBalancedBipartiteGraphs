"""
Dataset generation for stable matching experiments.
Based on WeaveNet paper: https://arxiv.org/abs/2310.12515
"""

import numpy as np
import torch
import pandas as pd
import os
from typing import Tuple, List, Dict
from dataclasses import dataclass
import pickle

@dataclass
class MatchingInstance:
    """Represents a stable matching problem instance"""
    N: int  # Number of agents on side A
    M: int  # Number of agents on side B
    preference_A: np.ndarray  # Shape: (N, M) - rankings from A to B
    preference_B: np.ndarray  # Shape: (M, N) - rankings from B to A
    satisfaction_A: np.ndarray  # Shape: (N, M) - normalized scores
    satisfaction_B: np.ndarray  # Shape: (M, N) - normalized scores
    distribution_type: str

class StableMatchingDataGenerator:
    """Generate synthetic datasets for stable matching experiments"""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def _rank_to_satisfaction(self, preferences: np.ndarray, N: int, C_min: float = 0.1) -> np.ndarray:
        """
        Convert preference rankings to normalized satisfaction scores.
        Based on equation (11) in the paper.

        Args:
            preferences: Array of rankings (1 to N)
            N: Problem size
            C_min: Minimum satisfaction value

        Returns:
            Normalized satisfaction scores in (C_min, 1.0]
        """
        # h(p_ij) = ((1 - C_min)(N - p_ij))/N + C_min
        satisfaction = ((1 - C_min) * (N - preferences)) / N + C_min
        return satisfaction

    def generate_uniform_preferences(self, N: int, M: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate uniform random preferences (UU distribution)"""
        # Each agent's preference is totally random from U(0,1)
        pref_scores_A = self.rng.uniform(0, 1, (N, M))
        pref_scores_B = self.rng.uniform(0, 1, (M, N))

        # Convert scores to rankings (1 = most preferred)
        pref_A = np.zeros((N, M), dtype=int)
        pref_B = np.zeros((M, N), dtype=int)

        for i in range(N):
            pref_A[i] = np.argsort(-pref_scores_A[i]) + 1

        for j in range(M):
            pref_B[j] = np.argsort(-pref_scores_B[j]) + 1

        return pref_A, pref_B

    def generate_discrete_preferences(self, N: int, M: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate discrete bimodal preferences (DD distribution)"""
        num_popular = max(1, int(0.4 * N))  # 40% are popular candidates

        pref_A = np.zeros((N, M), dtype=int)
        pref_B = np.zeros((M, N), dtype=int)

        # Generate for side A
        for i in range(N):
            scores = np.zeros(M)
            # Popular candidates get scores from U(0.5, 1)
            scores[:num_popular] = self.rng.uniform(0.5, 1, num_popular)
            # Others get scores from U(0, 0.5)
            scores[num_popular:] = self.rng.uniform(0, 0.5, M - num_popular)
            pref_A[i] = np.argsort(-scores) + 1

        # Generate for side B
        for j in range(M):
            scores = np.zeros(N)
            scores[:num_popular] = self.rng.uniform(0.5, 1, num_popular)
            scores[num_popular:] = self.rng.uniform(0, 0.5, N - num_popular)
            pref_B[j] = np.argsort(-scores) + 1

        return pref_A, pref_B

    def generate_gaussian_preferences(self, N: int, M: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate Gaussian-distributed preferences (GG distribution)"""
        pref_A = np.zeros((N, M), dtype=int)
        pref_B = np.zeros((M, N), dtype=int)

        # Generate for side A
        for i in range(N):
            # Each agent's preference toward j-th candidate ~ N(j/M, 0.4)
            scores = np.array([self.rng.normal(j/M, 0.4) for j in range(1, M+1)])
            pref_A[i] = np.argsort(-scores) + 1

        # Generate for side B
        for j in range(M):
            scores = np.array([self.rng.normal(i/N, 0.4) for i in range(1, N+1)])
            pref_B[j] = np.argsort(-scores) + 1

        return pref_A, pref_B

    def generate_mixed_preferences(self, N: int, M: int, dist_A: str, dist_B: str) -> Tuple[np.ndarray, np.ndarray]:
        """Generate mixed distribution preferences (e.g., UD)"""
        generators = {
            'U': self.generate_uniform_preferences,
            'D': self.generate_discrete_preferences,
            'G': self.generate_gaussian_preferences
        }

        if dist_A == 'U' and dist_B == 'D':
            # Side A has uniform preferences
            pref_scores_A = self.rng.uniform(0, 1, (N, M))
            pref_A = np.zeros((N, M), dtype=int)
            for i in range(N):
                pref_A[i] = np.argsort(-pref_scores_A[i]) + 1

            # Side B has discrete preferences
            num_popular = max(1, int(0.4 * N))
            pref_B = np.zeros((M, N), dtype=int)
            for j in range(M):
                scores = np.zeros(N)
                scores[:num_popular] = self.rng.uniform(0.5, 1, num_popular)
                scores[num_popular:] = self.rng.uniform(0, 0.5, N - num_popular)
                pref_B[j] = np.argsort(-scores) + 1

            return pref_A, pref_B
        else:
            # Fallback to uniform for both sides
            return self.generate_uniform_preferences(N, M)

    def generate_libimSeTi_preferences(self, N: int, M: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate LibimSeTi-like preferences based on real dating data distribution.
        This is a simplified version - full implementation would need the actual dataset.
        """
        # Simplified version: biased toward certain rating patterns
        # Real implementation would use the 2D distribution from LibimSeTi data

        pref_A = np.zeros((N, M), dtype=int)
        pref_B = np.zeros((M, N), dtype=int)

        # Create correlated preferences (more realistic)
        for i in range(N):
            # Some agents are generally more/less attractive
            base_attractiveness = self.rng.uniform(0.2, 0.8, M)
            noise = self.rng.normal(0, 0.3, M)
            scores = np.clip(base_attractiveness + noise, 0, 1)
            pref_A[i] = np.argsort(-scores) + 1

        for j in range(M):
            base_attractiveness = self.rng.uniform(0.2, 0.8, N)
            noise = self.rng.normal(0, 0.3, N)
            scores = np.clip(base_attractiveness + noise, 0, 1)
            pref_B[j] = np.argsort(-scores) + 1

        return pref_A, pref_B

    def generate_instance(self, N: int, M: int, distribution_type: str) -> MatchingInstance:
        """Generate a single matching instance"""

        if distribution_type == "UU":
            pref_A, pref_B = self.generate_uniform_preferences(N, M)
        elif distribution_type == "DD":
            pref_A, pref_B = self.generate_discrete_preferences(N, M)
        elif distribution_type == "GG":
            pref_A, pref_B = self.generate_gaussian_preferences(N, M)
        elif distribution_type == "UD":
            pref_A, pref_B = self.generate_mixed_preferences(N, M, 'U', 'D')
        elif distribution_type == "Lib":
            pref_A, pref_B = self.generate_libimSeTi_preferences(N, M)
        else:
            raise ValueError(f"Unknown distribution type: {distribution_type}")

        # Convert to satisfaction scores
        sat_A = self._rank_to_satisfaction(pref_A, M)
        sat_B = self._rank_to_satisfaction(pref_B, N)

        return MatchingInstance(
            N=N, M=M,
            preference_A=pref_A,
            preference_B=pref_B,
            satisfaction_A=sat_A,
            satisfaction_B=sat_B,
            distribution_type=distribution_type
        )

    def generate_dataset(self, sizes: List[int], distributions: List[str],
                        num_samples: int, split: str = "train") -> List[MatchingInstance]:
        """Generate a complete dataset"""
        dataset = []

        for N in sizes:
            M = N  # Square instances
            for dist_type in distributions:
                print(f"Generating {num_samples} samples for N={N}, M={M}, dist={dist_type}, split={split}")

                for _ in range(num_samples):
                    instance = self.generate_instance(N, M, dist_type)
                    dataset.append(instance)

        return dataset

    def save_dataset(self, dataset: List[MatchingInstance], filepath: str):
        """Save dataset to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"Saved {len(dataset)} instances to {filepath}")

    def load_dataset(self, filepath: str) -> List[MatchingInstance]:
        """Load dataset from file"""
        with open(filepath, 'rb') as f:
            dataset = pickle.load(f)
        print(f"Loaded {len(dataset)} instances from {filepath}")
        return dataset


def create_datasets():
    """Create all datasets needed for WeaveNet experiments"""

    generator = StableMatchingDataGenerator(seed=42)

    # Dataset configurations from paper
    sizes_small = [3, 5, 7, 9]  # For comparison with learning baselines
    sizes_medium = [20, 30]      # For comparison with algorithmic baselines
    sizes_large = [100]          # For scalability demonstration

    distributions = ["UU", "DD", "GG", "UD", "Lib"]

    # Generate validation and test sets (1000 samples each)
    for split in ["val", "test"]:
        for size_group, sizes in [("small", sizes_small), ("medium", sizes_medium), ("large", sizes_large)]:
            dataset = generator.generate_dataset(sizes, distributions, 1000, split)
            generator.save_dataset(dataset, f"data/{size_group}_{split}.pkl")

    # Training data is generated on-the-fly during training (too large to store)
    print("Dataset generation complete!")


if __name__ == "__main__":
    create_datasets()
