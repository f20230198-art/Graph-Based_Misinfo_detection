from __future__ import annotations

import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(RESULTS, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

MODEL_ORDER = ["RF_motif", "TextOnly_BERT", "GraphOnly_GAT", "PropNet_Fusion"]
SLICE_ORDER = ["pooled", "upfd_gos", "upfd_pol"]
MODEL_COLORS = {
    "RF_motif":       "#8c564b",
    "TextOnly_BERT":  "#ff7f0e",
    "GraphOnly_GAT":  "#9467bd",
    "PropNet_Fusion": "#2ca02c",
}

sns.set_context("talk")
sns.set_style("whitegrid")


def _save(fig, name: str) -> None:
    path = os.path.join(FIG_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def _preds_for(model: str, slice_: str = "pooled") -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(RESULTS, "predictions", f"{model}_s*.csv")))
    if not files:
        return pd.DataFrame()
    parts = []
    for f in files:
        seed = int(re.search(r"_s(\d+)\.csv$", f).group(1))
        df = pd.read_csv(f)
        df["seed"] = seed
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    if slice_ != "all":
        df = df[df["slice"] == slice_] if slice_ != "pooled" else df
    return df


def plot_metric_bars(summary_csv: str) -> None:
    if not os.path.exists(summary_csv):
        print(f"  skip metric_bars: {summary_csv} missing")
        return
    agg = pd.read_csv(summary_csv)
    metric_specs = [("roc_auc", "ROC-AUC"), ("f1_macro", "F1 (macro)"), ("balanced_accuracy", "Balanced Accuracy")]
    fig, axes = plt.subplots(1, 3, figsize=(22, 6), sharey=False)
    x = np.arange(len(MODEL_ORDER))
    width = 0.26
    for ax, (col, title) in zip(axes, metric_specs):
        for i, slice_ in enumerate(SLICE_ORDER):
            sub = agg[agg["slice"] == slice_].set_index("model")
            means = [sub.loc[m, f"{col}_mean"] if m in sub.index else np.nan for m in MODEL_ORDER]
            stds  = [sub.loc[m, f"{col}_std"]  if m in sub.index else 0.0     for m in MODEL_ORDER]
            ax.bar(x + (i - 1) * width, means, width, yerr=stds, capsize=4, label=slice_)
        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_ORDER, rotation=20, ha="right")
        ax.set_title(title)
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y", alpha=0.3)
    axes[-1].legend(title="slice", loc="lower right")
    fig.suptitle("Per-model performance across slices (mean ± std over seeds)", y=1.03)
    _save(fig, "metric_bars.png")


def plot_roc_curves() -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    for model in MODEL_ORDER:
        df = _preds_for(model, slice_="pooled")
        if df.empty:
            continue
        fpr, tpr, _ = roc_curve(df["y_true"], df["y_prob"])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{model} (AUC={roc_auc:.3f})",
                color=MODEL_COLORS.get(model), linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves — pooled test (all seeds concatenated)")
    ax.legend(loc="lower right", fontsize=11)
    _save(fig, "roc_curves_pooled.png")


def plot_confusion_matrices() -> None:
    fig, axes = plt.subplots(len(MODEL_ORDER), len(SLICE_ORDER), figsize=(13, 4 * len(MODEL_ORDER)))
    if len(MODEL_ORDER) == 1:
        axes = np.array([axes])
    for i, model in enumerate(MODEL_ORDER):
        for j, slice_ in enumerate(SLICE_ORDER):
            ax = axes[i, j]
            df = _preds_for(model, slice_=slice_)
            if df.empty:
                ax.set_visible(False)
                continue
            cm = confusion_matrix(df["y_true"], df["y_pred"], labels=[0, 1])
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                        xticklabels=["real", "fake"], yticklabels=["real", "fake"], ax=ax)
            ax.set_title(f"{model} — {slice_}")
            ax.set_xlabel("pred")
            ax.set_ylabel("true")
    fig.suptitle("Confusion matrices (pooled over seeds)", y=1.001)
    _save(fig, "confusion_matrices.png")


def plot_training_curves() -> None:
    gnn_models = ["GraphOnly_GAT", "PropNet_Fusion"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for model in gnn_models:
        files = sorted(glob.glob(os.path.join(RESULTS, "histories", f"{model}_s*.csv")))
        if not files:
            continue
        for k, f in enumerate(files):
            df = pd.read_csv(f)
            lbl = model if k == 0 else None
            axes[0].plot(df["epoch"], df["train_loss"], color=MODEL_COLORS.get(model), alpha=0.55, label=lbl)
            if "val_roc_auc" in df.columns:
                axes[1].plot(df["epoch"], df["val_roc_auc"], color=MODEL_COLORS.get(model), alpha=0.55, label=lbl)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("train loss"); axes[0].set_title("Training loss per seed")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("val ROC-AUC"); axes[1].set_title("Validation AUC per seed")
    for ax in axes:
        ax.legend(fontsize=11); ax.grid(alpha=0.3)
    _save(fig, "training_curves.png")


def plot_rf_importance() -> None:
    files = sorted(glob.glob(os.path.join(RESULTS, "predictions", "rf_importance_s*.csv")))
    if not files:
        print("  skip rf_importance: no files")
        return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    agg = df.groupby("feature")["importance"].agg(["mean", "std"]).reset_index().sort_values("mean", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(agg["feature"], agg["mean"], xerr=agg["std"], color="#8c564b")
    ax.set_xlabel("RF feature importance (mean ± std over seeds)")
    ax.set_title("Motif feature importance (Random Forest baseline)")
    _save(fig, "rf_motif_importance.png")


def plot_fusion_alpha() -> None:
    files = sorted(glob.glob(os.path.join(RESULTS, "predictions", "fusion_alpha_s*.csv")))
    if not files:
        print("  skip fusion_alpha: no files")
        return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].hist(df["alpha"], bins=40, color=MODEL_COLORS["PropNet_Fusion"], edgecolor="white")
    axes[0].axvline(0.5, color="k", linestyle="--", alpha=0.5)
    axes[0].set_xlabel("alpha (gate weight on GNN stream)")
    axes[0].set_ylabel("count")
    axes[0].set_title("Attention-gate alpha distribution (test graphs, all seeds)")
    axes[0].set_xlim(0, 1)

    for slice_ in SLICE_ORDER[1:]:
        sub = df[df["slice"] == slice_]
        if not sub.empty:
            axes[1].hist(sub["alpha"], bins=30, alpha=0.55, label=slice_, edgecolor="white")
    axes[1].axvline(0.5, color="k", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("alpha")
    axes[1].set_title("alpha by dataset slice")
    axes[1].legend()
    axes[1].set_xlim(0, 1)
    _save(fig, "fusion_alpha_hist.png")


def plot_sanity_shuffle() -> None:
    path = os.path.join(RESULTS, "sanity_label_shuffle.csv")
    if not os.path.exists(path):
        print("  skip sanity_shuffle: no file")
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [MODEL_COLORS.get(m, "#555") for m in df["model"]]
    ax.bar(df["model"], df["auc_shuffled"], color=colors)
    ax.axhline(0.5, color="k", linestyle="--", label="chance")
    ax.axhspan(0.45, 0.55, alpha=0.15, color="green", label="chance band")
    ax.set_ylim(0, 1)
    ax.set_ylabel("test AUC (labels shuffled)")
    ax.set_title("Label-shuffle sanity check: all models collapse to chance — no leakage")
    ax.set_xticklabels(df["model"], rotation=20, ha="right")
    ax.legend()
    _save(fig, "sanity_label_shuffle.png")


def main():
    print(f"Writing figures to {FIG_DIR}")
    plot_metric_bars(os.path.join(RESULTS, "upfd_multiseed_summary.csv"))
    plot_roc_curves()
    plot_confusion_matrices()
    plot_training_curves()
    plot_rf_importance()
    plot_fusion_alpha()
    plot_sanity_shuffle()
    print("Done.")


if __name__ == "__main__":
    main()
