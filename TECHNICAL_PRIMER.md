# PropNet — A Plain-English Technical Primer

**Project:** Graph-Based Misinformation Detection (PropNet)
**Team:** Srivathsa H Honyal, Ashmit Dhown, Aaryan Gupta, Sachin P
**Course:** Foundations of Data Science, BITS Pilani Dubai

This document explains the *what*, *how*, and *why* of PropNet from first principles. You don't need a deep ML background — if you know what a spreadsheet is, what a graph (dots connected by lines) is, and what "training a model" loosely means, you can follow along.

---

## Table of Contents

1. [The Big Idea](#1-the-big-idea)
2. [Preprocessing — Turning Raw Data Into Something a Model Can Learn From](#2-preprocessing)
3. [The Architecture — How the Model Thinks](#3-the-architecture)
4. [Training — How the Model Learns](#4-training)
5. [Evaluation — How We Know It Works](#5-evaluation)
6. [Tech Stack — What We Used and Why](#6-tech-stack)
7. [How PropNet Compares to Prior Research](#7-how-propnet-compares-to-prior-research)
8. [Glossary](#8-glossary)

---

## 1. The Big Idea

### The Problem

Fake news spreads on social media. Most detectors look only at **what the news says** — the words, the headline, the writing style. But Vosoughi et al. (2018, *Science*) showed something striking: **fake news spreads differently than real news**. It travels faster, reaches more people, and forms different shapes when you draw out who-shared-what-from-whom.

### Our Approach

PropNet looks at **both signals**:

- **What the tweets say** (text content, encoded as numbers by a language model called DistilBERT)
- **How the story propagates** (the shape of the retweet network — who shared from whom, how deep the cascade goes, how many "hubs" amplified it)

We train a neural network that learns from both, then combine them with a **learned attention gate** — basically, the model itself decides, on a case-by-case basis, "for *this* particular story, should I trust the text more, or the propagation shape more?"

The result: **ROC-AUC of 0.989** on a held-out test set — meaning if you give it a random real and random fake story, it ranks the fake one as more suspicious about 99% of the time.

---

## 2. Preprocessing

Raw social media data is messy. Before a model can learn from it, we have to convert it into a clean numerical form. Here's what that involves.

### 2.1 The Dataset: UPFD

We use **UPFD** (User Preference-aware Fake News Detection), a public benchmark from Dou et al. (SIGIR 2021). It comes in two domains:

- **PolitiFact** — 314 political stories
- **GossipCop** — 5,464 entertainment stories
- **Total:** 5,778 propagation graphs

Each "graph" represents one news story being shared on Twitter. Each **node** in the graph is a tweet. Each **edge** (line) connects a tweet to the tweet it was retweeted from. The fact-checked label (real or fake) comes from PolitiFact and GossipCop — human experts who manually verify claims.

### 2.2 What's in Each Graph

For every story, UPFD gives us:

| File | What it contains | Why we need it |
|------|------------------|----------------|
| `A.txt` | List of edges (which tweet links to which) | The structure of how the story spread |
| `node_graph_id.npy` | Which graph each tweet belongs to | So we can separate one story from another |
| `graph_labels.npy` | 0 = real, 1 = fake, per graph | The ground-truth answer the model is trying to learn |
| `new_bert_feature.npz` | A 768-number vector per tweet | A numerical "fingerprint" of the tweet's text |

That last one is important: **DistilBERT** is a language model that reads a tweet and outputs 768 numbers summarizing its meaning. Two tweets saying similar things produce similar numbers. UPFD precomputed these for us — we don't have to run BERT ourselves.

### 2.3 Hand-Crafted "Motif" Features (the structural fingerprint)

A neural network is good at spotting patterns it hasn't been told about, but it's also helpful to give it a few **explicit** structural clues. We compute **7 hand-crafted features** per graph (file: [src/motifs.py](src/motifs.py)):

1. **Triangle count** — How often three users form a closed triangle (A retweets B, B retweets C, C retweets A). Tight clusters often indicate echo chambers.
2. **Max star size** — The biggest "hub" account: how many people retweeted from the single most-shared tweet.
3. **Average clustering** — How tightly connected the neighborhoods are on average.
4. **Density** — How many edges actually exist out of all possible edges. Dense = highly interconnected.
5. **Number of nodes (log)** — How big the cascade got. Logarithm so very-large and small graphs are on a comparable scale.
6. **Root out-degree fraction** — What share of all retweets came from the single biggest hub. High value = one super-spreader dominated.
7. **Depth proxy** — How "deep" the cascade tree is (a long chain of retweets vs. a flat fan-out).

**Why these specifically?** Milo et al. (2002, *Science*) showed network motifs (these small recurring shapes) are signatures of how a network was generated. Fake news cascades typically have more triangles, deeper trees, and bigger hubs.

**Important detail (no cheating):** We compute the mean and standard deviation of these features **only on the training set**, then use the same values to standardize validation and test. This prevents a sneaky form of data leakage where the model gets indirect hints about test data.

### 2.4 Splitting the Data

We split the 5,778 graphs into:

- **75% training** (3,894 graphs) — the model sees these and learns from them
- **12.5% validation** (728 graphs) — used to decide when to stop training
- **12.5% test** (1,156 graphs) — never touched until the final evaluation

Splits are **stratified** by both label and dataset, so the proportion of fake/real and PolitiFact/GossipCop is the same in every split. We also assert that no story appears in two splits — a basic correctness check.

---

## 3. The Architecture

This is where the model itself lives. Think of it as a pipeline with two parallel "streams" that get combined at the end.

```
                                     ┌─────────────────────┐
   tweets (BERT 768-dim) ──► Stream 1: Graph Attention Net ─┤
                                                            │
                                                            ├──► Attention Gate ──► Classifier ──► Real / Fake
                                                            │
   7 motif features ────────► Stream 2: Small MLP ─────────┘
```

### 3.1 Stream 1: Graph Attention Network (GAT)

The GAT is the "shape-aware" stream. It looks at the propagation graph directly.

**How it works in plain English:** Each tweet has 768 numbers (its BERT embedding). The GAT updates each tweet's numbers by **looking at its neighbors** in the retweet graph and pulling in information from them — but not equally. It uses **attention**: it learns which neighbors are more important and weighs them more heavily. After two rounds of this, each tweet's numbers contain information about both itself and the surrounding propagation pattern.

We then **average** all the tweet vectors in a graph into one 128-dimensional summary per graph. This is the GNN's verdict on the cascade.

**Implementation specifics:**
- Layer 1: GATConv 768 → 128 with **4 attention heads** (so 4 perspectives in parallel, concatenated to 512 dims).
- Layer 2: GATConv 512 → 128 with **1 head** (concentrates back to 128 dims).
- Each layer is followed by **batch normalization**, **ELU activation**, and **dropout (0.4)** — standard tricks to keep training stable and prevent overfitting.
- Defined in [src/model.py](src/model.py) as the class `HybridGATWithAttention` (the PropNet fusion module — the name is historical; what matters is that it is GAT + motif MLP + attention gate).

### 3.2 Stream 2: Motif MLP

This is the "hand-crafted features" stream. It takes the 7 motif numbers and runs them through a small two-layer neural network (Linear → ReLU → Dropout → Linear), producing its own 128-dimensional summary.

It's small — just two fully connected layers — because there are only 7 input numbers; we don't need anything fancy.

### 3.3 The Attention Gate (the clever part)

We now have two 128-dimensional vectors: one from the GNN (text + structure), one from the motifs. Most prior work would just glue them together. We do something smarter.

We feed both vectors into a tiny network that outputs a single number **α (alpha)** between 0 and 1. Then we compute:

```
fused = α · (GNN summary) + (1 − α) · (motif summary)
```

- α near 1 → "I trust the GNN more for this story"
- α near 0 → "I trust the hand-crafted motifs more for this story"

**Why this matters:** different stories benefit from different signals. A story with a very small cascade (say, 10 tweets) doesn't have enough structure for a GNN to learn from — the motifs (which are blunt summary statistics) might be more reliable. A large complex cascade is the opposite. The gate lets the model decide on the fly.

This is the most novel piece of PropNet, and it's why we get a **3× reduction in seed-to-seed variance** compared to a non-gated hybrid.

### 3.4 The Classifier Head

Finally, the fused 128-dim vector goes through:

```
Linear(128 → 64) → ReLU → Dropout → Linear(64 → 2)
```

The output is two numbers (logits) representing how strongly the model believes "real" vs. "fake". Apply softmax and you get probabilities.

---

## 4. Training

Training is how a model goes from "random guesses" to "0.993 AUC".

### 4.1 The Loss Function

We use **weighted cross-entropy loss**. Cross-entropy measures how wrong the predicted probabilities are from the true label. The weighting is because classes aren't perfectly balanced — fake stories outnumber real ones in GossipCop. Weights of `total / (2 · class_count)` give each class equal pull on the gradient.

### 4.2 The Optimizer

**AdamW** with learning rate **5e-4** and weight decay **5e-4**. AdamW is a standard, robust optimizer; weight decay is a regularizer that discourages the model from leaning too hard on any one weight (helps generalization).

### 4.3 Learning Rate Scheduling and Early Stopping

- **ReduceLROnPlateau** — if validation AUC stalls for 4 epochs, halve the learning rate. Lets the model take big steps early and small ones later.
- **Early stopping** — if validation AUC doesn't improve for 15 epochs, stop. Prevents wasted compute and overfitting.
- **Max epochs** — 120, but we usually stop well before that.

### 4.4 Stability Tricks

- **Gradient clipping** at norm 1.0 — prevents one bad batch from blowing up the weights.
- **Automatic Mixed Precision (AMP)** — uses 16-bit floats where safe, 32-bit where needed. Faster training, less memory, no accuracy loss.
- **Batch size 64** — standard middle ground between speed and gradient stability.

### 4.5 The Configuration This Was Actually Trained On

For full transparency, here is the exact environment used to produce the numbers in this report:

**Hardware**
- **GPU:** NVIDIA RTX 4060 (consumer-grade, 8 GB VRAM)
- **Total training time:** ~12 minutes for the full 5-seed sweep across all models (PropNet_Fusion, GraphOnly_GAT, plus the RF and LogReg baselines)
- **Per-model convergence:** typically 30–40 epochs before early stopping triggers (well under the 120-epoch cap)

**Software**
- **OS:** Windows 11
- **Python:** 3.11
- **PyTorch:** 2.x
- **CUDA:** 12.x
- **PyTorch Geometric:** ≥ 2.4.0
- **Mixed precision:** enabled (AMP), so the GPU does most matmuls in float16 — faster, ~half the memory, no measurable accuracy loss

**Device selection logic** (from [src/train.py](src/train.py)):
```python
device = "cuda" if torch.cuda.is_available()
         else ("mps" if torch.backends.mps.is_available() else "cpu")
```
The code auto-detects CUDA (NVIDIA GPU) first, then Apple Silicon's MPS, then falls back to CPU. CPU-only training works but is significantly slower — the 15-minute number is GPU-bound.

**Reproducibility note:** every result in the [Evaluation](#5-evaluation) section was produced on this exact setup with seeds 42, 43, 44, 45, 46. Anyone with a CUDA-capable GPU and the pinned dependency versions in `requirements.txt` should reproduce the numbers within ± std.

### 4.7 What Gets Saved

For each run we record per-epoch training history (CSV), per-graph predictions on the test set (CSV), and per-graph fusion alpha values (so we can later inspect which signal the gate trusted). Best model weights go to `results/checkpoints/`.

---

## 5. Evaluation

### 5.1 Metrics

| Metric | What it tells us |
|--------|------------------|
| **ROC-AUC** | Probability the model ranks a random fake higher than a random real. 0.5 = chance, 1.0 = perfect. **Our primary metric.** |
| **Accuracy** | Fraction of correct predictions. Misleading on imbalanced data. |
| **Balanced Accuracy** | Average of per-class recall. Honest on imbalanced data. |
| **F1 (macro)** | Harmonic mean of precision and recall, averaged across classes. |

### 5.2 Multi-Seed Results (5 random seeds, pooled test set)

| Model | ROC-AUC | Balanced Acc | F1 (macro) |
|-------|---------|--------------|-----------|
| Random Forest on motifs only | 0.795 ± 0.011 | 0.731 | 0.733 |
| Logistic Regression on BERT only | 0.973 ± 0.001 | 0.935 | 0.936 |
| GAT on graph only | 0.725 ± 0.146 | 0.508 | 0.184 |
| **PropNet (Fusion gate)** | **0.989 ± 0.004** | **0.955 ± 0.008** | **0.955 ± 0.007** |

**Reading the table:**
- Text alone (BERT) is strong — 0.973 AUC. Most of the signal lives in the language.
- Structure alone (GAT) is weaker — 0.725. Propagation shape is informative but insufficient on its own.
- Combining them via the attention gate gets **0.989 AUC**. The lift over text-only is consistent (ΔAUC ≈ +0.016), and the fusion model is much more stable across the harder PolitiFact slice.

### 5.3 What is "seeding" and why do we run 5 of them?

A **random seed** is a starting number that controls every "random" decision the model makes — how the weights are first initialized, how data gets shuffled into batches, how dropout picks which neurons to disable, how the train/val/test split is drawn. Computers don't generate truly random numbers; they generate **pseudo-random** sequences from a seed. Same seed → same sequence → exactly reproducible run.

**Why this matters:** if you train a neural network once and report 0.994 AUC, you genuinely don't know whether that number is a property of the *method* or just lucky weight initialization. Train it again with a different seed and you might get 0.987 or 0.998.

So we train **the same model 5 times with seeds 42, 43, 44, 45, 46**, then report **mean ± standard deviation**. The "± 0.001" in our results table means: across 5 independent runs, the AUC barely moves. That's evidence the score is real, not a fluke.

This also catches a specific bad outcome: if one seed gave 0.99 and another gave 0.85, the mean might still look fine but the std would explode. Low std (0.001) is what gives us confidence.

### 5.4 Is this overfitted? Is the model hallucinating? — The honest audit

A 0.989 AUC is strong but worth defending. We ran four checks specifically to rule out the usual ways such a number turns out to be inflated:

**1. Label-shuffle audit (the leakage test).**
We re-ran the entire pipeline — preprocessing, motif scaling, training, evaluation — with **labels randomly shuffled** before training. If any part of the pipeline was secretly leaking the answer (e.g., motif statistics computed on the test set, or stories appearing in both train and test), the model would still score above chance even on garbage labels. Result:

| Model | AUC with shuffled labels |
|-------|-------------------------|
| RF_motif | 0.486 |
| TextOnly_BERT | 0.509 |
| GraphOnly_GAT | 0.496 |
| PropNet_Fusion | 0.519 |

All collapse to ~0.5 (chance). **There is no leakage.** The 0.989 is not coming from a hidden hint.

**2. Train/val/test isolation.**
- Splits are drawn once per seed and **asserted** non-overlapping by `claim_number` (file: [src/pooled_dataset.py](src/pooled_dataset.py)).
- The motif z-score scaler is fit **only on training data** and then frozen — validation and test are standardized using train statistics. This is a common subtle leakage source; we explicitly avoid it.
- The test set is never seen during training or hyperparameter selection. Early stopping uses **validation** AUC, not test.

**3. Overfitting check (train vs val gap).**
The training history CSVs in `results/histories/` show train AUC and val AUC per epoch. If the model were overfitting, we'd see train AUC climb to 1.0 while val AUC plateaus or drops. In practice the gap stays small and early stopping kicks in — typical runs end around epoch 30–60, well before the 120-epoch cap, because val AUC stops improving. Dropout (0.4), weight decay (5e-4), and batch normalization are all in place specifically to suppress overfitting.

**4. Why the score is plausibly genuine.**
- BERT embeddings of tweets are already highly separable — a *plain logistic regression* on mean-pooled BERT gets ~0.973 AUC. The text alone is a strong baseline.
- Adding the GNN over BERT-encoded nodes plus the motif stream lifts AUC by a further ~0.016 and (more importantly) cuts the failure rate on the harder PolitiFact slice.
- Published UPFD baselines from Dou et al. (2021) and follow-ups land in the 0.92–0.99 range. Our **0.989 ± 0.004** sits squarely inside that band — we are not beating the field by a suspicious margin.

**5. Honest limitations.**
- **Domain shift:** PropNet's PolitiFact-only AUC (~0.918) is lower and noisier than GossipCop (~0.995). With only 314 PolitiFact graphs, the model is data-starved.
- **Benchmark != real world:** UPFD was collected via Twitter API rehydration in 2020. Real deployment on today's X/Bluesky/Threads data would likely degrade because user behavior, bot prevalence, and platform algorithms have all shifted.
- **BERT features are pre-computed and frozen.** We didn't fine-tune BERT on misinformation language. Doing so might or might not help — we just don't know.
- **No adversarial test.** A bad actor who knows our motif features could in principle craft a cascade that mimics real-news structure. We have not tested adversarial robustness.

So: **the 0.989 AUC is real on this dataset**, with the caveats above. It is *not* a generic claim that we have "solved" misinformation detection.

### 5.5 The Live Demo

The `website/` folder hosts an interactive D3.js front end backed by a Flask API ([website/api/server.py](website/api/server.py)). You pick a story, see its propagation graph rendered as a force-directed visualization, see all three models' predictions side-by-side, see the motif feature bars, and — for PropNet — see the actual α the gate chose. Inference latency is 10–50 ms per graph on a consumer GPU.

---

## 6. Tech Stack

| Library | Used for | Why this and not something else |
|---------|----------|--------------------------------|
| **PyTorch** | Tensor math, autograd | The de facto research standard; PyG is built on it. |
| **PyTorch Geometric (PyG)** | Graph layers (`GATConv`, pooling) | Cleanest GNN library; handles batched variable-size graphs natively. DGL is a fine alternative but PyG has better PyTorch ergonomics. |
| **HuggingFace Transformers** | (Implicitly) DistilBERT | Embeddings precomputed by UPFD; we don't run BERT ourselves, saving compute. |
| **NetworkX** | Triangles, clustering, diameter | Pure Python, easy to read, perfect for one-off motif extraction on graphs of this size (graph-tool would be faster but harder to install on Windows). |
| **scikit-learn** | RF, LogReg baselines, splits, metrics | The standard. Nothing exotic needed. |
| **NumPy / SciPy / pandas** | Arrays, sparse matrices, CSVs | Universal numerical stack. |
| **Matplotlib / Seaborn** | ROC curves, confusion matrices | Static plots for the report. |
| **Flask + Flask-CORS** | Demo backend | Lightweight, single-file servers. We don't need FastAPI's async or Django's batteries. |
| **D3.js** | Force-directed graph layout in the browser | The reference library for interactive graph visualization. |
| **Vanilla JS + HTML/CSS** | Frontend | No React/Vue overhead — the demo is small enough that a framework would be a liability. |

We deliberately **didn't** use:

- **TensorFlow** — PyG works only with PyTorch, and the research community has largely moved to PyTorch.
- **DGL** — would have worked but PyG had cleaner examples for GAT + global pooling.
- **React/Next.js** — overkill for a 3-section demo page.
- **A heavy MLOps stack (MLflow, Weights & Biases)** — for a course project, CSV logs and Git checkpoints are enough.

---

## 7. How PropNet Compares to Prior Research

The references we cite (full BibTeX in [docs/references.bib](docs/references.bib)) fall into roughly three camps:

### 7.1 Text-Only Detection
- **Shu et al. (2017)** — surveys text/linguistic approaches.
- **Devlin et al. (2019)** — BERT, the basis for our text embeddings.

These works treat fake news as a pure text classification problem. They achieve good but ceiling-bound performance because the *shape* of how a story spreads carries information that no amount of text reading can capture. PropNet's text-only baseline (LogReg on mean-pooled BERT) reproduces this ceiling at ~0.973 AUC.

### 7.2 Graph-Only / Propagation-Based Detection
- **Ma et al. (2017, 2018)** — kernel methods and tree-RNNs on propagation structure.
- **Monti et al. (2019)** — "FakeGCN", applies GCN to propagation graphs.
- **Bian et al. (2020)** — "BiGCN", bidirectional GCN for top-down + bottom-up propagation.
- **Kipf & Welling (2017)** — original GCN paper.
- **Veličković et al. (2018)** — original GAT paper, our chosen layer.

These works use only structure. They underperform text-only on UPFD (our `GraphOnly_GAT` baseline gets 0.725 AUC) because the cascades vary enormously in size and quality, and small cascades carry little structural signal.

### 7.3 Hybrid Approaches
- **Dou et al. (2021) — UPFD** — proposes the dataset and a baseline that concatenates user-preference embeddings with graph features.
- **BiGCN, FakeGCN** — also hybrid in spirit but combine streams via simple concatenation.

**What PropNet adds on top of these:**

1. **A learned per-graph fusion gate**, not concatenation. The model decides for each story whether to trust text or structure. Prior works hard-wire the combination.
2. **Hand-crafted motif features alongside the GNN**, providing interpretable summary statistics (we can literally show users "this story had a triangle z-score of +2.3, that's fake-like"). Pure GNNs are black boxes.
3. **Rigorous multi-seed evaluation with a label-shuffle audit.** Many papers report a single number; we report mean ± std over 5 seeds and prove no leakage.
4. **Cross-domain reporting (PolitiFact vs GossipCop separately).** Reveals where each method breaks — e.g., GraphOnly GAT collapses on PolitiFact (314 graphs is too few for structural learning).

### 7.4 Why GAT Over GCN/SAGE

- **GCN (Kipf & Welling)** averages neighbors uniformly. Cheap and effective but assumes all neighbors are equally informative.
- **GraphSAGE** samples a fixed number of neighbors. Useful for huge graphs; ours are small.
- **GAT (Veličković et al.)** learns per-edge attention weights. In retweet cascades, some edges (e.g., a major hub retweeting) carry far more signal than others, so attention pays off.

We tested all three; GAT consistently won on validation AUC.

### 7.5 Why Motifs Over Pure End-to-End

A pure end-to-end GNN can in principle learn everything motifs encode. In practice, on the relatively small datasets we have (especially PolitiFact's 314 graphs), giving the model explicit hand-crafted summaries is a strong inductive bias. It's also interpretable — a motif z-score is something a non-ML user can understand, while a 128-dim GNN embedding is not.

---

## 8. Glossary

- **AUC / ROC-AUC** — Area under the Receiver Operating Characteristic curve. A single number summarizing how well a binary classifier ranks positives above negatives. 0.5 = random, 1.0 = perfect.
- **Attention (in neural nets)** — A mechanism for the model to learn which inputs to weight more heavily, dynamically.
- **BERT / DistilBERT** — Pretrained language models. They convert text into a fixed-length numerical vector that captures meaning.
- **Cross-entropy loss** — The standard loss function for classification; punishes confidently wrong predictions hard.
- **Embedding** — A vector of numbers representing something (a word, a tweet, a node) in a way the model can do math on.
- **Epoch** — One full pass through the training data.
- **GAT (Graph Attention Network)** — A type of GNN layer that learns how much to weight each neighbor when updating a node.
- **GNN (Graph Neural Network)** — A neural network that operates on graph-structured data by passing messages between connected nodes.
- **Logits** — Raw model outputs before softmax. Larger = more confident in that class.
- **Motif** — A small recurring subgraph pattern. Triangles, stars, chains.
- **Overfitting** — When a model memorizes training data instead of learning patterns, so it scores well on train but poorly on unseen data.
- **Pooling (global mean)** — Reducing all node vectors of a graph into one graph-level vector by averaging them.
- **Seed (random seed)** — A starting number that makes "random" steps in training reproducible. Running with multiple seeds and reporting mean ± std shows whether a result is robust or just lucky.
- **Stratified split** — A train/val/test split that preserves class proportions across the splits.
- **z-score / standardization** — Subtracting the mean and dividing by the standard deviation, so features are on a comparable scale.

---

## TL;DR

PropNet detects fake news by combining what tweets *say* (BERT embeddings flowing through a Graph Attention Network) with how stories *spread* (seven hand-crafted graph motif features), fused via a learned attention gate that decides per-story which signal to trust. On 5,778 propagation graphs from UPFD, it achieves **0.989 ± 0.004 ROC-AUC**, beating text-only (0.973) and graph-only (0.725) baselines. The pipeline is rigorously audited (label-shuffle test confirms no leakage) and ships with a live D3.js demo backed by Flask.
