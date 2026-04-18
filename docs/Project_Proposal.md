# Graph-Based Social Network Analysis for Misinformation Detection

> **Course:** Foundations of Data Science (FDS)
> **Team:** Ashmit Dhown, Aaryan Gupta, Sachin P, Srivathsa H

---

## What This Project Is About (In Plain English)

Imagine someone shares a fake news article on social media. Within minutes, hundreds of accounts retweet it, comment on it, and spread it further. Most fake news detectors try to read the article and decide if the words sound fake. But clever misinformation is written to *look* real.

**Our approach is different.** Instead of only reading the text, we look at *how the post is spreading* across the network of users. We ask:

- **How fast** is it spreading? (Fake news often spreads in sudden bursts)
- **Who** is spreading it? (Often the same small group of accounts, or bots)
- **What does the network pattern look like?** (Fake news creates dense, suspicious clusters)

Think of it like detective work: instead of just reading a letter, you also check who delivered it, how many copies were made, and whether the same courier keeps showing up. That pattern tells you a lot.

---

## The Problem We Are Solving

Most existing fake news detection tools only analyze the **text content** of a post. This misses critical signals:

| What They Miss | Why It Matters |
|---|---|
| **Coordinated groups** | Organized clusters of accounts amplifying the same content |
| **Bot-like accounts** | Automated profiles designed to boost reach |
| **Influential hubs** | A few accounts controlling the entire spread |
| **Sudden rapid resharing** | Abnormal speed compared to how real news spreads |

**Real news vs. Fake news patterns:**

| Feature | Real News | Fake News |
|---|---|---|
| Spread speed | Gradual, organic | Rapid, bursty |
| Who shares it | Diverse, unrelated users | Same group over and over |
| Network shape | Spread out, tree-like | Dense clusters, few sources broadcasting |
| Control | Distributed among many | Controlled by a few accounts |

---

## Our Approach: PropNet

We build a system called **PropNet** (Propagation-aware Network) that combines two types of analysis:

### Branch 1: Text / Semantic Branch
- Uses **BERT** (DistilBERT) to turn each tweet into a 768-dimensional meaning vector
- These vectors become the features on the nodes of the propagation graph

### Branch 2: Graph / Propagation Branch
- Models social media as a **graph**:
  - **Nodes** = tweets
  - **Edges** = propagation links (who retweeted/replied to whom)
- A **Graph Attention Network (GAT)** learns patterns over this graph
- A **motif module** computes recurring subgraph signals (triangles, star hubs, cascade depth)

### Fusion: Combining Both
A learnable **attention gate** decides, for each graph, how much to trust the GNN embedding vs. the motif signal. The final output is a probability: how likely is this post to be misinformation?

```
Tweets ──> BERT ──> Node features ──> GAT ──> Graph embedding ┐
                                                              ├──> Attention gate ──> Real / Fake
Graph structure ──> Motif extractor ──> Motif features ───────┘
```

---

## The Four Novelty Claims (from our presentation)

| # | Novelty | How We Deliver It |
|---|---|---|
| 1 | **Temporal Graphs** — track how cascades evolve and accelerate | Inter-event timing + cascade depth as node-level features |
| 2 | **GNN + BERT Fusion** — combine structure with semantic understanding | GAT over BERT-encoded tweet nodes, fused with motif stream via learned gate |
| 3 | **Graph Motifs** — detect coordinated inauthentic behavior via recurring subgraph patterns | Per-graph motif vector: triangles, max star size, clustering, density, depth, hub-dominance |
| 4 | **Cross-Domain Testing** — validate across domains | Pooled training + per-domain evaluation on PolitiFact (politics) and GossipCop (entertainment) |

---

## Dataset

### UPFD (User Preference-aware Fake News Detection)

UPFD is the graph-structured release of **FakeNewsNet**, distributed through the `safe-graph/GNN-FakeNews` repository and PyTorch Geometric. Each graph is a full retweet propagation tree with DistilBERT features already computed per node.

| Detail | Value |
|---|---|
| **PolitiFact** | 314 news graphs (politics) |
| **GossipCop** | 5,464 news graphs (entertainment) |
| **Node features** | 768-d DistilBERT per tweet (precomputed by UPFD authors) |
| **Edges** | Retweet / reply propagation links |
| **Ground truth** | Fact-checker verified (PolitiFact, GossipCop) |

**Why UPFD is the right fit:** the original FakeNewsNet download pipeline requires Twitter API rehydration, which was discontinued in 2023 after Twitter removed free academic API access. UPFD provides the same underlying FakeNewsNet propagation data in a ready-to-train graph format, avoiding that blocker entirely.

### Download

```python
from torch_geometric.datasets import UPFD
UPFD(root="datasets/UPFD_Dataset", name="politifact", feature="bert")
UPFD(root="datasets/UPFD_Dataset", name="gossipcop", feature="bert")
```

Or clone the raw files from https://github.com/safe-graph/GNN-FakeNews into `datasets/UPFD_Dataset/politifact/raw/` and `datasets/UPFD_Dataset/gossipcop/raw/`.

### FibVID (health domain — cross-domain stress test only)

FibVID (1,353 COVID-19 claims, 221K tweets) is kept as an **out-of-domain stress test** rather than part of the main training set. The model's behavior on FibVID is reported as a cross-domain generalization finding.

---

## Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| **Text encoding** | DistilBERT (HuggingFace) | 768-d tweet embeddings (via UPFD precomputed features) |
| **Graph Neural Network** | GAT (PyTorch Geometric) | Learns propagation patterns |
| **Motif extraction** | NetworkX | Triangles, star size, clustering, depth, density |
| **Attention fusion** | PyTorch | Learned gate between GNN and motif streams |
| **Baselines** | scikit-learn | Random Forest + Logistic Regression |
| **Visualization** | Matplotlib | ROC curves, confusion matrices, training plots |

---

## How We Evaluate Success

### Metrics
| Metric | What It Measures |
|---|---|
| **ROC-AUC** | How well the model ranks fake vs. real (primary metric) |
| **Balanced accuracy** | Average recall across classes — robust to class imbalance |
| **F1 (macro)** | Harmonic mean of precision and recall, averaged across classes |
| **Per-domain slicing** | Metrics computed separately on PolitiFact and GossipCop |

### Ablation Plan
We test three model configurations on UPFD to show each component adds value:

| Model | What It Tests |
|---|---|
| **Hybrid GAT** | GAT over BERT node features — the GNN-only baseline |
| **PropNet Fusion** | Hybrid GAT + motif stream + attention gate — the full system |

### Robustness
- **Multi-seed training** — every final number is reported across multiple random seeds
- **Stratified pooled split** — train/val/test split preserves label × domain ratios
- **Cross-domain reporting** — PolitiFact and GossipCop metrics reported separately
- **FibVID as out-of-domain probe** — quantifies domain-shift behavior

---

## Literature Review

We critically analyzed 3 key papers and identified their weaknesses:

### Paper 1: Graph-Theory SEIR Health Misinformation in Aging Populations
*Orugboh, Ezeogu & Juba, MJH 2025*
- SEIR model parameters lack proper validation
- Dataset not described (size, timeframe, how age was inferred)
- Simulated interventions never tested against real-world policies
- Older adults treated as one generic group
- Ethics and privacy concerns not addressed

### Paper 2: Knowledge-Based Fake News Detection with BERT + GCN/GNN
*Arshad, Manzoor & Hassan, MJH 2025*
- The GNN part (the main claimed contribution) was never actually tested — only BERT results shown
- Twitter dataset poorly described and not publicly available
- No ablation study to justify each component's value
- Multimodal features only described conceptually, not implemented

### Paper 3: Contextual + Sentiment + Social-Credibility Hybrid
*Bellam & Aakula, IEEE AIC 2025*
- Frequently misclassifies satire as fake news
- Cannot handle image/video misinformation
- Complex ensemble architecture with no explainability tooling
- Only tested on English news datasets

---

## References

1. *Graph Neural Network for Fake News Detection in Social Media* — arXiv 2502.16157
2. *A Graph Network Approach to Disinformation Detection* — ResearchGate 2025
3. *Context-Based Fake News Detection (COVID-19 Use-case)* — ACM DL
4. *Fake news detection: A survey of GNN methods* — PMC
5. *Misinformation detection based on news dispersion* — IEEE
6. Gangireddy et al., *Unsupervised Fake News Detection (GTUT)*, 2020
7. Spanakis group, *Fake News Detection via Propagation Patterns*, 2020
8. Hiremath et al., *Analysis of Fake News Detection using GNNs*, IEEE 2023
9. Gul et al., *Advancing fake news detection with GNN and deep learning*, 2025
10. Orugboh, Ezeogu & Juba, *Graph theory approach to health misinformation in aging populations*, MJH 2025
11. Arshad, Manzoor & Hassan, *Fake news detection via BERT + knowledge-aware GNN*, MJH 2025
12. Bellam & Aakula, *Contextual and Sentiment-Based Hybrid Models for Detecting Fake News*, IEEE AIC 2025
13. UPFD dataset — [GNN-FakeNews](https://github.com/safe-graph/GNN-FakeNews)
14. FakeNewsNet — [GitHub](https://github.com/KaiDMML/FakeNewsNet) (source data for UPFD)

---

*Foundations of Data Science — Project Proposal*
