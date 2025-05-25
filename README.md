# On Fair and Balanced Matching in Bipartite Graphs

*Honors Academy – Artificial Intelligence Track @ TU/e, September 2024 – May 2025*

---

### Project snapshot

| Item                  | Detail                                                                                                                       |
| --------------------- |------------------------------------------------------------------------------------------------------------------------------|
| **Student**           | Beloslava Malakova                                                                                                           |
| **Scientific mentor** | Dr. Thiago Simão (TU/e)                                                                                                      |
| **Honors track**      | Artificial Intelligence                                                                                                      |
| **Timeline**          | Kick-off 1 Sep 2024 -> hand-in 25th of May 2025                                                                              |
| **Goal**              | Compare graph-neural surrogates with Gale–Shapley, Serial Dictatorship and ACDA on fair, balanced student–college matchings. |

---

### Repository layout

```
fair-matching/
├ classical_algs.py          # GS, SD, ACDA
├ data_structs.py            # dense / sparse generators + loaders
├ data_struct2.py            # strict-mutual sparse generator
│
├ weavenet.py                # manual WeaveNet
├ weavenetlayers.py          # PyG WeaveNet
│
├ train_experiment.py        # training + HP search
├ fairness_eval.py           # EF-k, ESD, ARD, WLR
│
├ demo_dense.py              # dense pipeline
├ demo_sparse.py             # sparse pipeline
├ bar_chart.py               # EF-0 bar chart
│
└ data/                      # JSON preference files (runtime)
```

---

### Environment setup (CPU, Python 3.10)

```bash
conda create -n fairmatch python=3.10 -y
conda activate fairmatch
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric matplotlib pandas networkx
```

*(Install CUDA wheels first if you want GPU.)*

---

### Quick start

```bash
# 1. generate datasets
python data_structs.py

# 2. dense experiment
python demo_dense.py | tee dense_out.txt

# 3. sparse experiment
python demo_sparse.py | tee sparse_out.txt


```

---

### Code overview

| File                                | Purpose                                      |
| ----------------------------------- | -------------------------------------------- |
| `classical_algs.py`                 | Gale–Shapley, Serial Dictatorship, ACDA      |
| `data_structs.py`                   | dense / sparse generators & loaders          |
| `data_struct2.py`                   | mutual-edge sparse generator                 |
| `weavenet.py`                       | original WeaveNet                            |
| `weavenetlayers.py`                 | PyG MessagePassing WeaveNet                  |
| `train_experiment.py`               | `simple_train_loop`, `hyperparam_experiment` |
| `fairness_eval.py`                  | EF-k, ESD, ARD, WLR metrics                  |
| `demo_dense.py`, `s.demo_sparse.py` | end-to-end pipelines                         |

---

### Experimental settings

| Parameter          | Dense          | Sparse         |
| ------------------ | -------------- | -------------- |
| Students           | 500            | 500            |
| Colleges           | 50             | 50             |
| Capacity/college   | 15 (total 750) | 15 (total 750) |
| Edge density       | 100 %          | ≈ 30 % mutual  |
| Unmatched students | 0              | 0              |

---

### Neural-network configuration

* **Architecture:** WeaveNet, 4 layers, hidden 32
* **Loss:** Smooth-L1 on observed ranks (edges with fallback 100 skipped)
* **Optimisers tried:** SGD, Adam, AdamW, RMSprop
* **Training:** 10 epochs, full-graph batch

> *Future option:* swap in a differentiable fairness surrogate (soft EF-k) once a linear-time version is coded.

---


### Extending the project

* Add soft EF-k to the loss.
* Rank temperature scaling or Sinkhorn normalisation.
* Edge-attention WeaveNet (`GATConv`).
* Wider HP grid, real admissions datasets (e.g.\ NRMP).

---

### Contact & citation

If this repo helps your research, please cite
**B. Malakova, “On Fair and Balanced Matching in Bipartite Graphs,” TU/e Honors Academy, 2025.**
