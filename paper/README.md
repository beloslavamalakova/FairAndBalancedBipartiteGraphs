# Fair and Balanced Matching in Bipartite Graphs - README

## Overview

This repository contains a toy example showcasing:

- **Classical Matching Algorithms**: Gale-Shapley Deferred Acceptance, Serial Dictatorship, and ACDA.
- **Fairness and Evaluation**: Envy-freeness (EF-k), Exponential Score Difference (ESD), Average Rank Difference (ARD), and Weighted Logarithmic Rank (WLR).
- **Graph Neural Network (GNN) Approach**: A WeaveNet-style GNN in PyTorch Geometric, trained with a naive, partial approach for assigning students to colleges.

The code is split into multiple Python files for clarity:

1. **`data_structs.py`**  
   Holds data definitions and the global `STUDENT_PREFERENCES` dictionary. Contains a function `generate_synthetic_data(...)` to create toy student and college data.

2. **`classical_algs.py`**  
   Implements the Gale-Shapley Deferred Acceptance, Serial Dictatorship, and Artificial Cap Deferred Acceptance (ACDA) algorithms.

3. **`fairness_eval.py`**  
   Provides functions to compute EF-k (envy-freeness up to k) and the rank-based fairness metrics (ESD, ARD, WLR).

4. **`weavenet.py`**  
   Defines a WeaveNet-style Graph Neural Network in PyTorch Geometric, which processes bipartite graphs.

5. **`train_experiment.py`**  
   Contains the illustrative training loops, hyper-parameter search, and a helper function for turning GNN edge scores into a (greedy) matching.

6. **`run_demo.py`**  
   A driver script that puts it all together:
   - Generates synthetic data.
   - Runs Gale-Shapley, measuring fairness metrics.
   - Builds a PyTorch Geometric Data object for the bipartite graph.
   - Trains the GNN (naively, with non-differentiable matching).
   - Performs a small hyper-parameter experiment.

---

## Getting Started

1. **Clone or download** this repository into a local folder.
2. **Install dependencies** (Python 3.8+ recommended):
   ```bash
   pip install torch torch_geometric
   ```
   And any other relevant packages.

   





