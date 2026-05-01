"""
PropNet demo backend.

Loads UPFD test graphs + all 5 trained checkpoints once at startup, runs live
inference on demand, and returns rich JSON for the frontend Explorer.

Run from repo root:
    python website/api/server.py
"""
from __future__ import annotations

import os
import random
import sys
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
from flask import Flask, jsonify, request
from flask_cors import CORS

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.model import GATGraphClassifier, HybridGATWithAttention
from src.motifs import (
    MOTIF_DIM,
    MOTIF_FEATURE_NAMES,
    apply_motif_scaler,
    attach_motif_features,
    compute_motif_features,
    fit_motif_scaler,
)
from src.pooled_dataset import concat_datasets, stratified_pooled_split
from src.upfd_loader import load_upfd_graphs

CONFIG_PATH = os.path.join(ROOT, "configs", "config.yaml")
SEED = 42
DEFAULT_NEIGHBORS = 4  # max edges to render per node so the canvas stays readable


# ──────────────────────────────────────────────────────────────────────────────
# Boot-time data + model loading (heavy, runs once)
# ──────────────────────────────────────────────────────────────────────────────
def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _build_model(kind: str, input_dim: int, hidden_dim: int, dropout: float):
    if kind == "fusion":
        return HybridGATWithAttention(
            input_dim=input_dim, motif_dim=MOTIF_DIM,
            hidden_dim=hidden_dim, dropout=dropout,
        )
    return GATGraphClassifier(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)


def _structural_nodes(g):
    """Replicates predict.py — converts BERT features to degree-only features
    for the GraphOnly_GAT model."""
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


def _bootstrap():
    print("[boot] loading config…")
    cfg = _load_config()
    pol_root = os.path.join(ROOT, cfg["paths"]["upfd_politifact_root"])
    gos_root = os.path.join(ROOT, cfg["paths"]["upfd_gossipcop_root"])
    ckpt_dir = os.path.join(ROOT, cfg["paths"]["checkpoint_dir"])

    print(f"[boot] loading UPFD politifact + gossipcop…")
    pol = load_upfd_graphs(pol_root, dataset_id="upfd_pol")
    gos = load_upfd_graphs(gos_root, dataset_id="upfd_gos")
    pooled = concat_datasets(pol, gos)
    print(f"[boot]   total graphs: {len(pooled)} (pol={len(pol)}, gos={len(gos)})")

    print(f"[boot] computing motif features…")
    pooled = attach_motif_features(pooled)

    print(f"[boot] reproducing seed={SEED} stratified split…")
    train, val, test = stratified_pooled_split(
        pooled,
        test_size=cfg["split"]["test_size"],
        val_size=cfg["split"]["val_size"],
        random_state=SEED,
    )
    motif_mean, motif_std = fit_motif_scaler(train)
    apply_motif_scaler(train, motif_mean, motif_std)
    apply_motif_scaler(val, motif_mean, motif_std)
    apply_motif_scaler(test, motif_mean, motif_std)
    print(f"[boot]   test split: {len(test)} graphs")

    # Save raw motif features for human-readable display (un-standardized).
    raw_motif: Dict[str, np.ndarray] = {}
    for g in test:
        raw = compute_motif_features(g)  # recompute pre-scaling
        raw_motif[g.claim_number] = raw

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[boot] device: {device}")

    input_dim = int(test[0].x.shape[1])
    hidden_dim = cfg["training"]["hidden_dim"]
    dropout = cfg["training"].get("dropout", 0.4)

    print(f"[boot] loading checkpoints…")
    models: Dict[str, Any] = {}
    for kind, ckpt_name in [
        ("fusion",   f"ums_fusion_s{SEED}.pt"),
        ("hybrid",   f"ums_hybrid_s{SEED}.pt"),
        ("graphonly", f"ums_go_s{SEED}.pt"),
    ]:
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"[boot]   WARN: {ckpt_path} missing, skipping {kind}")
            continue
        in_dim = 2 if kind == "graphonly" else input_dim
        m = _build_model(kind, in_dim, hidden_dim, dropout).to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        m.load_state_dict(state)
        m.eval()
        models[kind] = m
        print(f"[boot]   loaded {kind} ← {ckpt_name}")

    if "fusion" not in models:
        raise RuntimeError("Fusion checkpoint missing — cannot run demo.")

    # Pre-render structural variants of the test graphs for graphonly model.
    test_struct = [_structural_nodes(g) for g in test]

    print("[boot] ready.\n")
    return {
        "test": test,
        "test_struct": test_struct,
        "raw_motif": raw_motif,
        "models": models,
        "device": device,
        "motif_mean": motif_mean,
        "motif_std": motif_std,
    }


STATE = _bootstrap()


# ──────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def _score_one(model, graph, kind: str) -> dict:
    device = STATE["device"]
    g = graph.to(device)
    batch = torch.zeros(g.x.shape[0], dtype=torch.long, device=device)
    if kind == "fusion":
        # Compute the same internals as the model so we can extract alpha.
        import torch.nn.functional as F
        from torch_geometric.nn import global_mean_pool

        h = model.input_norm(g.x)
        h = model.gat1(h, g.edge_index)
        h = model.bn1(h); h = F.elu(h)
        h = model.gat2(h, g.edge_index)
        h = model.bn2(h); h = F.elu(h)
        h_graph = global_mean_pool(h, batch)
        m_feat = model.motif_mlp(g.motif_feat.to(device))
        cat = torch.cat([h_graph, m_feat], dim=1)
        alpha_t = model.gate(cat).squeeze(-1)
        fused = alpha_t.unsqueeze(-1) * h_graph + (1.0 - alpha_t.unsqueeze(-1)) * m_feat
        logits = model.classifier(fused)
        alpha = float(alpha_t.item())
    else:
        logits = model(g.x, g.edge_index, batch)
        alpha = None

    probs = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
    return {
        "p_real": float(probs[0]),
        "p_fake": float(probs[1]),
        "pred":   int(np.argmax(probs)),
        "alpha":  alpha,
    }


def _graph_to_json(g, raw_motif: np.ndarray, motif_mean, motif_std) -> dict:
    """Convert a torch_geometric graph into the {nodes, links} format the
    frontend D3 renderer expects, plus all the inspector data."""
    n = int(g.x.shape[0])

    # Build adjacency from edge_index (it is undirected, double-listed)
    edge_index = g.edge_index.cpu().numpy()
    adj: Dict[int, List[int]] = {i: [] for i in range(n)}
    edge_set = set()
    for i in range(edge_index.shape[1]):
        a, b = int(edge_index[0, i]), int(edge_index[1, i])
        if a == b:
            continue
        key = (min(a, b), max(a, b))
        if key in edge_set:
            continue
        edge_set.add(key)
        adj[a].append(b)
        adj[b].append(a)

    # BFS from node 0 (the root in UPFD) to assign depth + group.
    depths = {0: 0}
    queue = [0]
    while queue:
        nxt = []
        for u in queue:
            for v in adj[u]:
                if v not in depths:
                    depths[v] = depths[u] + 1
                    nxt.append(v)
        queue = nxt

    # Pick top hubs by degree at depth 1 for visual emphasis.
    nodes = []
    for i in range(n):
        d = depths.get(i, 1)
        if i == 0:
            group = "root"
        elif d == 1:
            group = "hub"
        else:
            group = "leaf"
        nodes.append({"id": i, "depth": d, "group": group})

    links = [{"source": a, "target": b} for (a, b) in edge_set]

    # Compute z-scores for the inspector motif bars (frontend already styles
    # bars assuming z-scores).
    raw = raw_motif
    mean = motif_mean.squeeze(0)
    std = motif_std.squeeze(0)
    z = (raw - mean) / np.where(std < 1e-6, 1.0, std)
    motif_payload = {}
    for name, raw_v, z_v in zip(MOTIF_FEATURE_NAMES, raw, z):
        motif_payload[name] = {"raw": float(raw_v), "z": float(z_v)}

    # Cascade depth proxy, useful as a top-line stat
    max_depth = max(depths.values()) if depths else 0

    return {
        "nodes": nodes,
        "links": links,
        "num_nodes": n,
        "num_edges": len(edge_set),
        "max_depth": int(max_depth),
        "motif": motif_payload,
    }


def _serve_graph(g, struct_g) -> dict:
    raw_motif = STATE["raw_motif"][g.claim_number]
    payload = _graph_to_json(g, raw_motif, STATE["motif_mean"], STATE["motif_std"])

    # Run all available models
    preds = {}
    fusion_res = _score_one(STATE["models"]["fusion"], g, "fusion")
    preds["PropNet_Fusion"] = {
        "p_fake":   fusion_res["p_fake"],
        "p_real":   fusion_res["p_real"],
        "alpha":    fusion_res["alpha"],
        "correct":  fusion_res["pred"] == int(g.y.item()),
    }
    if "hybrid" in STATE["models"]:
        r = _score_one(STATE["models"]["hybrid"], g, "hybrid")
        preds["Hybrid_GAT"] = {
            "p_fake":  r["p_fake"], "p_real": r["p_real"],
            "correct": r["pred"] == int(g.y.item()),
        }
    if "graphonly" in STATE["models"]:
        r = _score_one(STATE["models"]["graphonly"], struct_g, "graphonly")
        preds["GraphOnly_GAT"] = {
            "p_fake":  r["p_fake"], "p_real": r["p_real"],
            "correct": r["pred"] == int(g.y.item()),
        }

    label = int(g.y.item())
    payload.update({
        "id":            g.claim_number,
        "dataset":       "PolitiFact" if g.dataset_id == "upfd_pol" else "GossipCop",
        "label":         label,
        "label_name":    "FAKE" if label == 1 else "REAL",
        "predictions":   preds,
    })
    return payload


# ──────────────────────────────────────────────────────────────────────────────
# Flask app
# ──────────────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=os.path.join(ROOT, "website"), static_url_path="")
CORS(app)


@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.get("/api/health")
def health():
    return jsonify({
        "ok": True,
        "device": STATE["device"],
        "test_size": len(STATE["test"]),
        "models": list(STATE["models"].keys()),
    })


@app.get("/api/index")
def list_index():
    """List of all test-set graphs with a tiny summary, for the sidebar."""
    items = []
    for i, g in enumerate(STATE["test"]):
        items.append({
            "idx": i,
            "id": g.claim_number,
            "dataset": "PolitiFact" if g.dataset_id == "upfd_pol" else "GossipCop",
            "label": int(g.y.item()),
            "num_nodes": int(g.x.shape[0]),
        })
    return jsonify({"items": items, "n": len(items)})


@app.get("/api/sample")
def sample():
    """Return one test graph with full live inference.

    Query params:
      label  = 'fake' | 'real' | 'any'   (default any)
      idx    = explicit test-set index   (overrides label if given)
    """
    test = STATE["test"]
    test_struct = STATE["test_struct"]

    idx_param = request.args.get("idx")
    if idx_param is not None:
        try:
            i = int(idx_param)
        except ValueError:
            return jsonify({"error": "idx must be an integer"}), 400
        if not (0 <= i < len(test)):
            return jsonify({"error": f"idx out of range [0, {len(test)-1}]"}), 400
        return jsonify(_serve_graph(test[i], test_struct[i]))

    want = request.args.get("label", "any").lower()
    if want not in ("fake", "real", "any"):
        return jsonify({"error": "label must be fake|real|any"}), 400

    pool_indices = list(range(len(test)))
    if want == "fake":
        pool_indices = [i for i in pool_indices if int(test[i].y.item()) == 1]
    elif want == "real":
        pool_indices = [i for i in pool_indices if int(test[i].y.item()) == 0]

    if not pool_indices:
        return jsonify({"error": "no graphs match"}), 404

    i = random.choice(pool_indices)
    return jsonify(_serve_graph(test[i], test_struct[i]))


if __name__ == "__main__":
    print("[boot] starting Flask on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
