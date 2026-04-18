from __future__ import annotations

import os
from typing import List

import numpy as np
import torch
from scipy import sparse
from torch_geometric.data import Data


def _load_bert_features(path: str) -> np.ndarray:
    with np.load(path, allow_pickle=True) as npz:
        keys = set(npz.keys())
        if {"data", "indices", "indptr", "shape"}.issubset(keys):
            mat = sparse.csr_matrix(
                (npz["data"], npz["indices"], npz["indptr"]),
                shape=tuple(npz["shape"].tolist()),
            )
            return mat.toarray().astype(np.float32)
        if len(keys) == 1:
            only = next(iter(keys))
            return npz[only].astype(np.float32)
    raise ValueError(f"Could not decode BERT features from {path}")


def load_upfd_graphs(root_dir: str, dataset_id: str) -> List[Data]:
    edges_path = os.path.join(root_dir, "A.txt")
    labels_path = os.path.join(root_dir, "graph_labels.npy")
    node_graph_id_path = os.path.join(root_dir, "node_graph_id.npy")
    features_path = os.path.join(root_dir, "new_bert_feature.npz")

    for p in (edges_path, labels_path, node_graph_id_path, features_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"UPFD file missing: {p}")

    edges = np.loadtxt(edges_path, delimiter=",", dtype=np.int64)
    if edges.ndim == 1:
        edges = edges.reshape(-1, 2)

    labels = np.load(labels_path)
    labels = np.rint(labels).astype(np.int64)
    node_graph_id = np.load(node_graph_id_path).astype(np.int64)
    features = _load_bert_features(features_path)

    if features.shape[0] != node_graph_id.shape[0]:
        raise ValueError(
            f"UPFD mismatch: features has {features.shape[0]} rows but node_graph_id has "
            f"{node_graph_id.shape[0]} entries in {root_dir}"
        )

    num_graphs = int(labels.shape[0])
    graphs: List[Data] = []

    node_graphs_of_edges = node_graph_id[edges[:, 0]]
    edges_by_graph: dict[int, list[np.ndarray]] = {}
    for i in range(edges.shape[0]):
        g = int(node_graphs_of_edges[i])
        edges_by_graph.setdefault(g, []).append(edges[i])

    for graph_idx in range(num_graphs):
        node_mask = node_graph_id == graph_idx
        node_ids = np.where(node_mask)[0]
        if node_ids.size == 0:
            continue

        global_to_local = {int(g): i for i, g in enumerate(node_ids)}
        x = torch.from_numpy(features[node_ids]).float()

        local_edges: List[List[int]] = []
        for e in edges_by_graph.get(graph_idx, []):
            src, dst = int(e[0]), int(e[1])
            if src in global_to_local and dst in global_to_local:
                ls, ld = global_to_local[src], global_to_local[dst]
                local_edges.append([ls, ld])
                local_edges.append([ld, ls])

        if local_edges:
            edge_index = torch.tensor(local_edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        label = int(labels[graph_idx])
        if label not in (0, 1):
            raise ValueError(f"Expected binary label in {{0,1}}, got {label} for {dataset_id} graph {graph_idx}")

        data = Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.long))
        data.claim_number = f"{dataset_id}_{graph_idx}"
        data.dataset_id = dataset_id
        graphs.append(data)

    return graphs
