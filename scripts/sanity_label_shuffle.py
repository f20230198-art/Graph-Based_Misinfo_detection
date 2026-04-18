from __future__ import annotations

import argparse
import copy
import os
import sys

import numpy as np
import torch
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from torch_geometric.loader import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.pooled_dataset import concat_datasets, stratified_pooled_split, split_by_dataset
from src.upfd_loader import load_upfd_graphs
from src.motifs import MOTIF_DIM, attach_motif_features, fit_motif_scaler, apply_motif_scaler
from src.train import TrainingConfig, train_gnn, evaluate_gnn
from src.utils import set_seed


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _shuffle_labels(graphs, seed: int):
    rng = np.random.default_rng(seed)
    labels = np.array([int(g.y.item()) for g in graphs])
    perm = rng.permutation(len(labels))
    shuffled = labels[perm]
    for g, yl in zip(graphs, shuffled):
        g.y = torch.tensor([int(yl)], dtype=torch.long)
    return graphs


def _strip_text(graphs):
    from torch_geometric.utils import degree as pyg_degree
    out = []
    for g in graphs:
        g2 = copy.copy(g)
        n = g.x.shape[0]
        if g.edge_index.numel() > 0:
            deg = pyg_degree(g.edge_index[0], num_nodes=n).float()
        else:
            deg = torch.zeros(n, dtype=torch.float32)
        deg_norm = deg / (deg.max() + 1e-6)
        g2.x = torch.stack([deg_norm, torch.ones_like(deg_norm)], dim=1)
        out.append(g2)
    return out


def _motif_matrix(graphs):
    X = np.stack([g.motif_feat.squeeze(0).numpy() for g in graphs], axis=0).astype(np.float32)
    y = np.array([int(g.y.item()) for g in graphs], dtype=np.int64)
    return X, y


def _text_matrix(graphs):
    X = np.stack([g.x.mean(dim=0).numpy() for g in graphs], axis=0).astype(np.float32)
    y = np.array([int(g.y.item()) for g in graphs], dtype=np.int64)
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_epochs", type=int, default=20)
    args = ap.parse_args()

    cfg = _load_config(args.config)
    set_seed(args.seed)

    print("=" * 60)
    print(f"LABEL-SHUFFLE SANITY CHECK (seed={args.seed}, max_epochs={args.max_epochs})")
    print("Expectation: every model's AUC should be ~0.5 (+/- ~0.05)")
    print("=" * 60)

    pol_root = cfg["paths"]["upfd_politifact_root"]
    gos_root = cfg["paths"]["upfd_gossipcop_root"]
    upfd_pol = load_upfd_graphs(pol_root, dataset_id="upfd_pol")
    upfd_gos = load_upfd_graphs(gos_root, dataset_id="upfd_gos")
    pooled = concat_datasets(upfd_pol, upfd_gos)
    pooled = attach_motif_features(pooled)
    print(f"Loaded {len(pooled)} graphs. Shuffling labels...")
    pooled = _shuffle_labels(pooled, seed=args.seed)

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
    print(f"Split: train/val/test = {len(train)}/{len(val)}/{len(test)}")

    t = cfg["training"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def gnn_cfg(ckpt, attn=False):
        return TrainingConfig(
            batch_size=t["batch_size"], hidden_dim=t["hidden_dim"],
            dropout=t.get("dropout", 0.4), lr=t["lr"], weight_decay=t["weight_decay"],
            max_epochs=args.max_epochs, patience=5,
            lr_scheduler_patience=3, lr_scheduler_factor=0.5,
            grad_clip_norm=1.0, use_amp=t["use_amp"],
            use_attention_fusion=attn,
            motif_dim=MOTIF_DIM if attn else 0,
            checkpoint_path=f"results/checkpoints/sanity_{ckpt}.pt",
        )

    results = {}

    Xtr, ytr = _motif_matrix(train); Xte, yte = _motif_matrix(test)
    rf = RandomForestClassifier(n_estimators=200, random_state=args.seed,
                                class_weight="balanced", n_jobs=-1).fit(Xtr, ytr)
    from sklearn.metrics import roc_auc_score
    results["RF_motif"] = roc_auc_score(yte, rf.predict_proba(Xte)[:, 1])

    Xtr, ytr = _text_matrix(train); Xte, yte = _text_matrix(test)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced",
                            random_state=args.seed).fit(Xtr, ytr)
    results["TextOnly_BERT"] = roc_auc_score(yte, lr.predict_proba(Xte)[:, 1])

    tr_go, va_go, te_go = _strip_text(train), _strip_text(val), _strip_text(test)
    go_res = train_gnn(tr_go, va_go, te_go, input_dim=tr_go[0].x.shape[1],
                       config=gnn_cfg("graphonly"))
    results["GraphOnly_GAT"] = go_res["metrics"]["roc_auc"]

    node_dim = train[0].x.shape[1]
    hy_res = train_gnn(train, val, test, input_dim=node_dim, config=gnn_cfg("hybrid"))
    results["Hybrid_GAT"] = hy_res["metrics"]["roc_auc"]

    fu_res = train_gnn(train, val, test, input_dim=node_dim, config=gnn_cfg("fusion", attn=True))
    results["PropNet_Fusion"] = fu_res["metrics"]["roc_auc"]

    print("\n" + "=" * 60)
    print("SANITY RESULTS (AUC on shuffled labels)")
    print("=" * 60)
    verdicts = []
    for name, auc_val in results.items():
        if 0.40 <= auc_val <= 0.60:
            verdict = "OK (chance-level)"
        elif 0.60 < auc_val <= 0.70:
            verdict = "SUSPICIOUS"
        else:
            verdict = "LEAK SUSPECTED"
        verdicts.append(verdict)
        print(f"  {name:16s} AUC={auc_val:.3f}   {verdict}")

    os.makedirs("results", exist_ok=True)
    import csv
    with open("results/sanity_label_shuffle.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "auc_shuffled", "verdict"])
        for (name, auc_val), v in zip(results.items(), verdicts):
            w.writerow([name, f"{auc_val:.4f}", v])
    print("\nSaved results/sanity_label_shuffle.csv")

    bad = [n for n, v in zip(results.keys(), verdicts) if v != "OK (chance-level)"]
    if bad:
        print(f"\nFAILED for: {bad}. Investigate leakage.")
        sys.exit(1)
    print("\nAll models at chance. Pipeline passes label-shuffle sanity check.")


if __name__ == "__main__":
    main()
