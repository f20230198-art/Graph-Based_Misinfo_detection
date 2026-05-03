# PropNet — Live Demo Run Sheet

Quick reference for demo day. The website is now backed by a Flask server that
loads the real `ums_fusion_s42.pt` checkpoint and runs PropNet inference live
on graphs from the held-out UPFD test split.

## One-time install

From the repo root:

```powershell
pip install -r requirements.txt
```

(Adds `flask` and `flask-cors` on top of the existing training stack.)

## Start the demo

From the repo root:

```powershell
python website/api/server.py
```

Wait for `[boot] ready.` (takes ~30 seconds — UPFD load + motif features +
checkpoint load), then open <http://127.0.0.1:5000> in a browser.

The first prediction loads automatically. Use the buttons in the sidebar:

- **Random FAKE** — pulls a random fake-labelled test graph
- **Random REAL** — pulls a random real-labelled test graph
- **Any** — uniform random
- **Test-set browse** list — pick a specific graph by ID

## What the prof sees (live)

For every graph clicked:

- D3 force-directed propagation graph (real edges, real depths)
- **Confidence card** — verdict, confidence %, p(real), p(fake), α gate value, latency
- α-stream bar showing how much PropNet relied on graph stream vs motif stream
- 7 standardised motif features with z-scores
- Per-model prediction comparison (GraphOnly + PropNet Fusion run side-by-side)
- Ground-truth label so prof can verify it's correct

## Static sections (already prepared)

- Ablation chart — uses aggregate stats from `results/upfd_multiseed_summary.csv`
- Results dashboard — six static plots from `assets/figures/`
- Integrity section — label-shuffle proof

## Troubleshooting

**`Checkpoint not found` at boot** → confirm `results/checkpoints/ums_fusion_s42.pt`
exists. If not, train a single seed: `python main.py --mode upfd_multiseed --seeds 42`.

**`Dataset path missing`** → confirm `datasets/politifact/raw/` and
`datasets/gossipcop/raw/` exist (each with `A.txt`, `graph_labels.npy`,
`node_graph_id.npy`, `new_bert_feature.npz`).

**Backend offline message in the sidebar** → make sure the Flask server is
running. The frontend talks to it on the same origin (port 5000).

**Slow first inference** → first call hits cold cuDNN kernels; subsequent
calls run in 10–50 ms on the RTX 4060.

## What to say if asked "is this live on Twitter?"

> "The model was trained on UPFD, the standard academic propagation-graph
> benchmark. Twitter's API is paid, so we can't pull arbitrary live tweets,
> but the same architecture works on any propagation graph with text-embedded
> nodes. What you're seeing here is real PropNet inference on the held-out
> test split — the model has never seen these graphs during training."
