import random
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_classification_metrics(y_true: Sequence[int], y_pred: Sequence[int], y_prob: Optional[Sequence[float]] = None) -> Dict[str, float]:
    y_true = list(y_true)
    y_pred = list(y_pred)

    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0, labels=[0, 1])
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0, labels=[0, 1])

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_real": float(per_class_precision[0]),
        "recall_real": float(per_class_recall[0]),
        "precision_fake": float(per_class_precision[1]),
        "recall_fake": float(per_class_recall[1]),
    }
    if y_prob is not None and len(set(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        metrics["roc_auc"] = float(auc(fpr, tpr))
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def plot_confusion(y_true, y_pred, title: str = "Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


def plot_roc(y_true, y_prob, title: str = "ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def visualize_graph(graph: nx.Graph, title: str = "Propagation Graph", max_nodes: int = 150):
    if graph.number_of_nodes() > max_nodes:
        sampled_nodes = list(graph.nodes())[:max_nodes]
        graph = graph.subgraph(sampled_nodes).copy()

    pos = nx.spring_layout(graph, seed=42)
    node_colors = []
    for n in graph.nodes():
        ntype = graph.nodes[n].get("node_type", 0)
        node_colors.append("#1f77b4" if ntype == 0 else "#ff7f0e")

    plt.figure(figsize=(9, 7))
    nx.draw_networkx_nodes(graph, pos, node_size=45, node_color=node_colors, alpha=0.85)
    nx.draw_networkx_edges(graph, pos, width=0.5, alpha=0.35)
    plt.title(title + " (blue=user, orange=post)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
