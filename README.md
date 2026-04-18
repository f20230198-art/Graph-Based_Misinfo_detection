# PropNet: Graph-Based Misinformation Detection

Hybrid Graph Neural Network that detects misinformation by combining **what a post says** (text) with **how it spreads** (propagation network).

**Team:** Srivathsa H Honyal, Ashmit Dhown, Aaryan Gupta, Sachin P
**Affiliation:** Dept. of Computer Science, BITS Pilani, Dubai Campus
**Course:** Foundations of Data Science

**Headline result (5-seed UPFD ablation, pooled test):** PropNet Fusion achieves **ROC-AUC 0.993 ± 0.001**, verified against a label-shuffle sanity check. Full results in [docs/Results_Report.tex](docs/Results_Report.tex).

---

## Quickstart

```bash
# 1. install
pip install -r requirements.txt

# 2. download UPFD into ./datasets/UPFD_Dataset/ (see "Datasets" below)

# 3a. single-seed run (Hybrid GAT + PropNet Fusion)
python main.py --mode upfd_only

# 3b. full ablation across seeds (RF + Text + GraphOnly + Hybrid + Fusion)
python main.py --mode upfd_multiseed --seeds 42 43 44 45 46

# 4. leakage sanity check (every model should collapse to ~0.5 AUC)
python scripts/sanity_label_shuffle.py --seed 42

# 5. render all report figures from dumps
python scripts/make_plots.py

# 6. CLI inference on a held-out test graph
python predict.py --sample-random --model fusion --seed 42
python predict.py --sample-index 0 --only fake --model fusion
```

Outputs:
- `results/upfd_multiseed_summary.csv` — mean ± std per (model, slice)
- `results/upfd_multiseed_runs.csv` — per-(seed, model, slice) rows (gitignored)
- `results/histories/*.csv` — per-epoch training curves per seed (gitignored)
- `results/predictions/*.csv` — per-graph test predictions + Fusion α + RF importance (gitignored)
- `results/figures/*.png` — 7 report figures (committed)
- `results/checkpoints/*.pt` — trained weights (gitignored)
- `results/sanity_label_shuffle.csv` — leakage audit

Older pipelines and pre-leak-fix results live in `archive/` (gitignored).

---

## Project layout

```
.
├── main.py                  # terminal entry point (train + ablate)
├── predict.py               # CLI inference against a trained checkpoint
├── configs/config.yaml      # all hyperparameters
├── src/                     # library code
│   ├── upfd_loader.py       # UPFD raw-file loader
│   ├── motifs.py            # graph-motif feature extraction + train-only scaler
│   ├── pooled_dataset.py    # stratified split by (label, dataset)
│   ├── model.py             # GAT + HybridGATWithAttention (motif fusion)
│   ├── train.py             # GNN training loop
│   └── utils.py             # metrics + plot helpers
├── scripts/
│   ├── sanity_label_shuffle.py   # legitimacy audit
│   └── make_plots.py             # renders all report figures
├── notebooks/               # exploratory end-to-end notebook
├── docs/                    # proposal, LaTeX report, references.bib
├── datasets/                # (gitignored) put UPFD here
└── results/
    ├── figures/             # 7 report figures (committed)
    ├── histories/           # (gitignored) per-epoch dumps
    ├── predictions/         # (gitignored) per-graph test dumps
    └── checkpoints/         # (gitignored) trained weights
```

---

## Datasets

UPFD goes under `./datasets/`. Not committed to git.

### UPFD (PolitiFact + GossipCop)
Pre-computed propagation graphs with DistilBERT node features. UPFD is the graph-structured release of FakeNewsNet; it side-steps the Twitter-API rehydration problem.

Easiest path — let PyG download it:
```python
from torch_geometric.datasets import UPFD
UPFD(root="datasets/UPFD_Dataset", name="politifact", feature="bert")
UPFD(root="datasets/UPFD_Dataset", name="gossipcop", feature="bert")
```

Or download raw files from https://github.com/safe-graph/GNN-FakeNews and place them at:
- `datasets/UPFD_Dataset/politifact/raw/`
- `datasets/UPFD_Dataset/gossipcop/raw/`

---

## Legitimacy

The pipeline includes three checks against inflated numbers:
1. **Train-only feature normalization** — motif scaler fit on train split, applied to val/test.
2. **Split-overlap assertion** — fails loudly if any `claim_number` appears in more than one split.
3. **Label-shuffle sanity check** — `scripts/sanity_label_shuffle.py` randomizes labels and re-runs the full ablation; every model must collapse to ~0.5 AUC. Passes with AUC ∈ [0.486, 0.519].

---

## Report

Compile the IEEE-format paper:
```bash
cd docs && pdflatex Results_Report.tex && bibtex Results_Report && pdflatex Results_Report.tex && pdflatex Results_Report.tex
```
