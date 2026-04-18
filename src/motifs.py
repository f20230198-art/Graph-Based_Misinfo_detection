from __future__ import annotations

from typing import List

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


MOTIF_FEATURE_NAMES: List[str] = [
    "triangle_count",
    "max_star_size",
    "avg_clustering",
    "density",
    "num_nodes_log",
    "root_out_degree_frac",
    "depth_proxy",
]
MOTIF_DIM = len(MOTIF_FEATURE_NAMES)


def compute_motif_features(data: Data) -> np.ndarray:
    feats = np.zeros(MOTIF_DIM, dtype=np.float32)
    num_nodes = int(data.x.shape[0]) if data.x is not None else 0
    if num_nodes == 0:
        return feats

    g = to_networkx(data, to_undirected=True)

    try:
        tri_total = sum(nx.triangles(g).values()) // 3
    except Exception:
        tri_total = 0
    feats[0] = float(np.log1p(tri_total))

    degrees = np.array([d for _, d in g.degree()], dtype=np.float32) if g.number_of_nodes() > 0 else np.array([0.0])
    feats[1] = float(np.log1p(float(degrees.max())))

    try:
        feats[2] = float(nx.average_clustering(g)) if g.number_of_nodes() > 1 else 0.0
    except Exception:
        feats[2] = 0.0

    feats[3] = float(nx.density(g)) if g.number_of_nodes() > 1 else 0.0

    feats[4] = float(np.log1p(num_nodes))

    total_edges = g.number_of_edges()
    if total_edges > 0 and degrees.size > 0:
        feats[5] = float(degrees.max() / (2.0 * total_edges))
    else:
        feats[5] = 0.0

    try:
        if g.number_of_nodes() > 1:
            comps = list(nx.connected_components(g))
            if comps:
                largest = g.subgraph(max(comps, key=len))
                if largest.number_of_nodes() > 1:
                    if largest.number_of_nodes() > 200:
                        src = next(iter(largest.nodes()))
                        lengths = nx.single_source_shortest_path_length(largest, src)
                        depth = max(lengths.values()) if lengths else 0
                    else:
                        depth = nx.diameter(largest)
                    feats[6] = float(np.log1p(depth))
    except Exception:
        feats[6] = 0.0

    return feats


def attach_motif_features(graphs: List[Data]) -> List[Data]:
    for g in graphs:
        mf = compute_motif_features(g)
        g.motif_feat = torch.from_numpy(mf).unsqueeze(0)
    return graphs


def fit_motif_scaler(graphs: List[Data]) -> tuple:
    if not graphs:
        return np.zeros((1, MOTIF_DIM), dtype=np.float32), np.ones((1, MOTIF_DIM), dtype=np.float32)
    stacked = np.stack([g.motif_feat.squeeze(0).numpy() for g in graphs], axis=0)
    mean = stacked.mean(axis=0, keepdims=True).astype(np.float32)
    std = stacked.std(axis=0, keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def apply_motif_scaler(graphs: List[Data], mean: np.ndarray, std: np.ndarray) -> List[Data]:
    for g in graphs:
        v = g.motif_feat.squeeze(0).numpy()
        normed = ((v - mean.squeeze(0)) / std.squeeze(0)).astype(np.float32)
        g.motif_feat = torch.from_numpy(normed).unsqueeze(0)
    return graphs


def normalize_motif_features(graphs: List[Data]) -> List[Data]:
    # Deprecated: fits on all graphs (leaks test stats). Use fit_motif_scaler(train) + apply_motif_scaler per split.
    if not graphs:
        return graphs
    mean, std = fit_motif_scaler(graphs)
    return apply_motif_scaler(graphs, mean, std)
