from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.motifs import MOTIF_DIM, MOTIF_FEATURE_NAMES, attach_motif_features, fit_motif_scaler, apply_motif_scaler
from src.model import GATGraphClassifier, HybridGATWithAttention
from src.pooled_dataset import concat_datasets, stratified_pooled_split
from src.upfd_loader import load_upfd_graphs


MODEL_KINDS = {
    "fusion":   {"ckpt_fmt": "ums_fusion_s{seed}.pt",  "needs_motif": True,  "input": "bert"},
    "graphonly":{"ckpt_fmt": "ums_go_s{seed}.pt",      "needs_motif": False, "input": "structural"},
}

LABEL_NAME = {0: "REAL", 1: "FAKE"}


def _load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _structural_nodes(g):
    import copy
    from torch_geometric.utils import degree as pyg_degree
    g2 = copy.copy(g)
    n = g.x.shape[0]
    if g.edge_index.numel() > 0:
        deg = pyg_degree(g.edge_index[0], num_nodes=n).float()
    else:
        deg = torch.zeros(n, dtype=torch.float32)
    deg_norm = deg / (deg.max() + 1e-6)
    g2.x = torch.stack([deg_norm, torch.ones_like(deg_norm)], dim=1)
    return g2


def _build_model(kind: str, input_dim: int, hidden_dim: int, dropout: float):
    if kind == "fusion":
        return HybridGATWithAttention(
            input_dim=input_dim, motif_dim=MOTIF_DIM,
            hidden_dim=hidden_dim, dropout=dropout,
        )
    return GATGraphClassifier(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)


def _score(model, graph, device: str, kind: str):
    model.eval()
    g = graph.to(device)
    batch = torch.zeros(g.x.shape[0], dtype=torch.long, device=device)

    TEMPERATURE = 1
    with torch.no_grad():
        if kind == "fusion":
            logits = model(g.x, g.edge_index, batch, g.motif_feat.to(device))
        else:
            logits = model(g.x, g.edge_index, batch)
        probs = torch.softmax(logits.float() / TEMPERATURE, dim=1)[0].cpu().numpy()
    pred = int(np.argmax(probs))
    return pred, float(probs[1]), float(probs[0])


def _print_graph_report(graph, pred, p_fake, p_real, model_name: str):
    true_label = int(graph.y.item())
    true_name = LABEL_NAME[true_label]
    pred_name = LABEL_NAME[pred]
    correct = pred == true_label

    print()
    print("=" * 60)
    print(f"  PREDICTION REPORT  |  model={model_name}")
    print("=" * 60)
    print(f"  graph id       : {graph.claim_number}")
    print(f"  dataset        : {graph.dataset_id}")
    print(f"  num_nodes      : {int(graph.x.shape[0])}")
    print(f"  num_edges      : {int(graph.edge_index.shape[1])}")
    if hasattr(graph, "motif_feat"):
        mf = graph.motif_feat.detach().cpu().squeeze(0).numpy()
        print(f"  motif features (standardized):")
        for name, val in zip(MOTIF_FEATURE_NAMES, mf):
            print(f"    {name:22s} {val:+.3f}")
    print("-" * 60)
    print(f"  p(real)  = {p_real:.4f}")
    print(f"  p(fake)  = {p_fake:.4f}")
    print(f"  predicted: {pred_name}")
    print(f"  actual   : {true_name}")
    mark = "CORRECT" if correct else "WRONG"
    print(f"  verdict  : {mark}")
    print("=" * 60)
    return correct


def main():
    ap = argparse.ArgumentParser(description="PropNet CLI inference on UPFD graphs.")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--model", choices=list(MODEL_KINDS.keys()), default="fusion")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--checkpoint", default=None)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--sample-random", action="store_true")
    src.add_argument("--sample-index", type=int)
    ap.add_argument("--only", choices=["real", "fake"], default=None)
    ap.add_argument("--n", type=int, default=1)
    args = ap.parse_args()

    cfg = _load_cfg(args.config)
    kind_info = MODEL_KINDS[args.model]

    ckpt_dir = cfg["paths"]["checkpoint_dir"]
    ckpt_path = args.checkpoint or os.path.join(ckpt_dir, kind_info["ckpt_fmt"].format(seed=args.seed))
    if not os.path.exists(ckpt_path):
        sys.exit(f"Checkpoint not found: {ckpt_path}\n"
                 f"Tip: run `python main.py --mode upfd_multiseed --seeds {args.seed}` first, "
                 f"or pass --checkpoint <path>.")

    pol_root = cfg["paths"]["upfd_politifact_root"]
    gos_root = cfg["paths"]["upfd_gossipcop_root"]
    for p in (pol_root, gos_root):
        if not os.path.exists(p):
            sys.exit(f"Dataset path missing: {p}")
    print(f"Loading UPFD (pol + gos)...")
    upfd_pol = load_upfd_graphs(pol_root, dataset_id="upfd_pol")
    upfd_gos = load_upfd_graphs(gos_root, dataset_id="upfd_gos")
    pooled = concat_datasets(upfd_pol, upfd_gos)
    pooled = attach_motif_features(pooled)

    # Reproduce the exact split used at training time.
    train, val, test = stratified_pooled_split(
        pooled,
        test_size=cfg["split"]["test_size"],
        val_size=cfg["split"]["val_size"],
        random_state=args.seed,
    )
    mean, std = fit_motif_scaler(train)
    apply_motif_scaler(train, mean, std)
    apply_motif_scaler(val, mean, std)
    apply_motif_scaler(test, mean, std)
    print(f"  split (seed={args.seed}): test={len(test)} held-out graphs")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if kind_info["input"] == "structural":
        test = [_structural_nodes(g) for g in test]

    input_dim = int(test[0].x.shape[1])
    hidden_dim = cfg["training"]["hidden_dim"]
    dropout = cfg["training"].get("dropout", 0.4)
    model = _build_model(args.model, input_dim, hidden_dim, dropout).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    print(f"  loaded {args.model.upper()} checkpoint: {ckpt_path}")

    rng = random.Random(args.seed)
    if args.sample_index is not None:
        if args.sample_index < 0 or args.sample_index >= len(test):
            sys.exit(f"--sample-index out of range [0, {len(test)-1}]")
        picks = [test[args.sample_index]]
    else:
        pool = test
        if args.only is not None:
            wanted = 1 if args.only == "fake" else 0
            pool = [g for g in test if int(g.y.item()) == wanted]
            if not pool:
                sys.exit(f"No graphs with true label '{args.only}' in test split.")
        n = min(args.n, len(pool))
        picks = rng.sample(pool, n)

    correct = 0
    for g in picks:
        pred, p_fake, p_real = _score(model, g, device, args.model)
        if _print_graph_report(g, pred, p_fake, p_real, args.model):
            correct += 1

    if len(picks) > 1:
        print(f"\nBatch accuracy: {correct}/{len(picks)} = {correct/len(picks):.1%}")


if __name__ == "__main__":
    main()
