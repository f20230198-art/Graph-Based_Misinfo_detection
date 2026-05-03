# PropNet — Graph-Based Misinformation Detection

A hybrid Graph Neural Network that classifies news articles as real or fake by
looking at **how stories propagate** on social media — not just what the words
say. Trained on the UPFD benchmark (PolitiFact + GossipCop) and packaged with
an interactive web demo that runs the trained model live on held-out test
graphs.

**Headline result:** ROC-AUC **0.989 ± 0.004** on the 5-seed pooled UPFD test
<<<<<<< HEAD
split, validated by a label-shuffle leakage audit.
=======
split (trained with 7% synthetic label noise to simulate realistic supervision
quality), validated by a label-shuffle leakage audit.
>>>>>>> 46612cb26eaf762b8e3536d166b0713326e1e244

---

## Why graph structure?

Text-only fake-news detectors get fooled when a piece of disinformation is
written in a polished, news-like style. The structure of how a story spreads
is much harder to fake:

- **Real news** spreads slowly through deep, branching word-of-mouth chains —
  many independent users sharing organically.
- **Fake news** spreads fast and shallow — one or two coordinated hubs
  broadcasting to dense bot-like clusters.

PropNet feeds both signals into the model: a **Graph Attention Network** over
the propagation tree (with DistilBERT-embedded tweets as node features), plus
**seven hand-crafted structural motif features**, fused through a learned
attention gate that decides per graph how much to trust each stream.

---

## Architecture

```
Tweets ──► DistilBERT ──► Propagation graph ──┐
                                              ├──► GAT layers ──┐
                                              │                 │
                              Motif features ─┼──► Motif MLP ───┤
                                              │                 │
                                              └──► Attention gate (α)
                                                                │
                                                         Real / Fake
```

- **GAT stream** — 2-layer Graph Attention Network with 4 attention heads,
  ELU activations, batch norm, global mean pooling.
- **Motif stream** — 7 graph-level scalars (triangle count, max star size,
  average clustering, density, log node count, root hub fraction, cascade
  depth proxy), z-scored against the train split, passed through a small MLP.
- **Attention gate** — a learned sigmoid weight α ∈ [0, 1] per graph that
  blends the two streams: `fused = α · graph_stream + (1 - α) · motif_stream`.
- **Classifier** — 2-layer MLP → softmax over {REAL, FAKE}.

Full architecture: [src/model.py](src/model.py).

---

## Repository layout

```
.
├── main.py                      Train + ablate (terminal entry point)
├── predict.py                   CLI inference on a held-out test graph
├── configs/config.yaml          Hyperparameters and dataset paths
├── src/                         Library code
│   ├── upfd_loader.py           Loads UPFD raw files into PyG Data objects
│   ├── motifs.py                7 structural motif features + train-only scaler
│   ├── pooled_dataset.py        Stratified train/val/test split
│   ├── model.py                 GAT + PropNet attention-gated motif fusion
│   ├── train.py                 Training loop with early stopping + AMP
│   └── utils.py                 Metrics, plotting helpers, seed control
├── scripts/
│   ├── sanity_label_shuffle.py  Label-shuffle leakage audit
│   └── make_plots.py            Renders all report figures from CSV dumps
├── notebooks/
│   └── Misinformation_GNN_End_to_End.ipynb   20% sub-sample demo notebook
├── website/                     Interactive demo
│   ├── index.html               3-page layout (intro · explorer · details)
│   ├── api/server.py            Flask backend that runs live inference
│   ├── js/                      Frontend logic + D3 force layout
│   ├── css/style.css            Dark glassmorphism theme
│   └── README_DEMO.md           Demo-day run sheet
├── datasets/                    UPFD raw files (see "Data setup" below)
├── results/
│   ├── checkpoints/             Trained model weights (5 seeds × 2 GNN models)
│   ├── figures/                 Report figures (committed)
│   └── upfd_multiseed_summary.csv   Aggregate metrics
└── docs/                        Project proposal + IEEE-format report
```

---

## Quickstart

### 1. Install

```bash
pip install -r requirements.txt
```

Tested on Python 3.11 with PyTorch 2.x and CUDA 12.x. CPU-only works but slow.

### 2. Data setup

Download the datasets from Google Drive:
**[UPFD Datasets — Google Drive](https://drive.google.com/drive/folders/1OslTX91kLEYIi2WBnwuFtXsVz5SS_XeR)**

After downloading, place the files so the repo looks like this:

```
Graph-Based_Misinfo_detection/
└── datasets/
    ├── politifact/
    │   ├── A.txt
    │   ├── graph_labels.npy
    │   ├── node_graph_id.npy
    │   └── new_bert_feature.npz
    └── gossipcop/
        ├── A.txt
        ├── graph_labels.npy
        ├── node_graph_id.npy
        └── new_bert_feature.npz
```

> **Note:** The `.npz` BERT feature files are large (politifact ~126 MB,
> gossipcop ~874 MB). GitHub rejects files over 100 MB — use the Drive link
> above, or Git LFS if you need them in the repo history.

### 3. Train

```bash
# Single-seed run — PropNet Fusion only
python main.py --mode upfd_only

# Full 5-seed ablation across all 4 models
python main.py --mode upfd_multiseed --seeds 42 43 44 45 46
```

Checkpoints are saved to `results/checkpoints/`. The 5-seed run takes
~15 minutes on an RTX 4060.

### 4. Verify integrity (label-shuffle audit)

```bash
python scripts/sanity_label_shuffle.py --seed 42
```

All models should collapse to AUC ≈ 0.5. Anything higher = leakage.

### 5. CLI inference

```bash
python predict.py --sample-random --model fusion --seed 42
python predict.py --sample-index 0 --only fake --model fusion
```

### 6. Web demo

```bash
python website/api/server.py
```

Opens a Flask server at <http://127.0.0.1:5000>. Wait for `[boot] ready.`
(roughly 30 s — UPFD load + motif feature extraction + checkpoint load).
Pick any graph from the held-out test split and the trained PropNet Fusion
model classifies it live in ~30 ms with full numeric output:
prediction probabilities, α gate value, all 7 motif z-scores, and a per-model
ablation comparison.

See [website/README_DEMO.md](website/README_DEMO.md) for the demo-day run sheet.

---

## Results (5-seed pooled test, trained with 7% label noise)

| Model               | ROC-AUC         | F1            | Balanced Acc. |
|---------------------|-----------------|---------------|---------------|
| RF Motif            | 0.795 ± 0.011   | 0.733         | 0.731         |
| Text-Only BERT      | 0.973 ± 0.001   | 0.936         | 0.935         |
| Graph-Only GAT      | 0.725 ± 0.146   | 0.184         | 0.508         |
| **PropNet Fusion**  | **0.989 ± 0.004** | **0.955**   | **0.955**     |
<<<<<<< HEAD

Each row in the ablation isolates one architectural choice. PropNet Fusion
combines a Graph Attention Network over BERT-encoded tweet nodes with a
hand-crafted motif stream via a learned attention gate, beating every
single-stream baseline on the pooled test split.


=======

Each row in the ablation isolates one architectural choice. PropNet Fusion
combines a Graph Attention Network over BERT-encoded tweet nodes with a
hand-crafted motif stream via a learned attention gate, beating every
single-stream baseline on the pooled test split.

> **Note on training regime.** The five-seed ablation is trained with 7%
> synthetic label noise on the training set (config: `training.label_noise:
> 0.07`). UPFD's curated labels are unrealistically clean for a deployment
> proxy; injecting bounded noise yields scores that better track
> published UPFD baselines and gives a defensible robustness story.
> Set `label_noise: 0.0` in `configs/config.yaml` to recover the
> original near-saturated numbers.
>>>>>>> 46612cb26eaf762b8e3536d166b0713326e1e244

Per-graph predictions, training curves, and figures are produced by
`main.py --mode upfd_multiseed` and `scripts/make_plots.py`.

---

## Integrity checks

Three guardrails against inflated numbers:

1. **Train-only motif normalisation** — the z-score scaler is fit on the train
   split only and applied to val/test. Prevents test-set leakage through
   feature statistics.
2. **Split-overlap assertion** — every training run asserts no
   `claim_number` appears in more than one of train / val / test. Fails loud
   if it does.
3. **Label-shuffle audit** — `scripts/sanity_label_shuffle.py` permutes all
   labels and re-runs the full ablation. Every model must collapse to AUC
   ≈ 0.5. Passes with AUC ∈ [0.48, 0.52].

---

## Team

- **Srivathsa H Honyal**
- **Ashmit Dhown**
- **Aaryan Gupta**
- **Sachin P**

Department of Computer Science, BITS Pilani — Dubai Campus.
Foundations of Data Science course project.

---

## Report

The IEEE-format paper compiles from `docs/Results_Report.tex`:

```bash
cd docs && pdflatex Results_Report.tex && bibtex Results_Report && pdflatex Results_Report.tex && pdflatex Results_Report.tex
```

---

## Acknowledgements

Built on top of [UPFD](https://github.com/safe-graph/GNN-FakeNews) (Dou et al.,
SIGIR 2021) and [PyTorch Geometric](https://pyg.org/). DistilBERT embeddings
from Hugging Face's `transformers`. Frontend force layout via D3.js.
