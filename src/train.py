from dataclasses import dataclass
import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader

from src.model import GATGraphClassifier, HybridGATWithAttention
from src.utils import compute_classification_metrics


@dataclass
class TrainingConfig:
    batch_size: int = 32
    hidden_dim: int = 128
    dropout: float = 0.4
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 120
    patience: int = 15
    lr_scheduler_patience: int = 4
    lr_scheduler_factor: float = 0.5
    min_delta: float = 1e-4
    grad_clip_norm: float = 1.0
    num_workers: int = 0
    use_amp: bool = True
    balance_datasets: bool = False
    use_attention_fusion: bool = False
    motif_dim: int = 0
    checkpoint_path: str = "results/checkpoints/best_gnn.pt"
    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    )


def _compute_class_weights(graphs, device: str) -> torch.Tensor:
    labels = [int(g.y.item()) for g in graphs]
    if len(labels) == 0:
        return torch.tensor([1.0, 1.0], dtype=torch.float32, device=device)

    counts = np.bincount(labels, minlength=2).astype(np.float32)
    total = float(counts.sum())
    weights = np.where(counts > 0, total / (2.0 * counts), 1.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _save_checkpoint(model: torch.nn.Module, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def _load_checkpoint_if_exists(model: torch.nn.Module, path: str) -> None:
    if os.path.exists(path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)


def _run_epoch(model, loader, optimizer, criterion, device: str, grad_clip_norm: float, use_amp: bool):
    model.train()
    total_loss = 0.0
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device == "cuda"))

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=(use_amp and device == "cuda")):
            if isinstance(model, HybridGATWithAttention):
                logits = model(batch.x, batch.edge_index, batch.batch, batch.motif_feat)
            else:
                logits = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(logits, batch.y)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

        total_loss += float(loss.item())

    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate_gnn(model, loader, device: str, use_amp: bool = True):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    for batch in loader:
        batch = batch.to(device)
        with torch.amp.autocast("cuda", enabled=(use_amp and device == "cuda")):
            if isinstance(model, HybridGATWithAttention):
                logits = model(batch.x, batch.edge_index, batch.batch, batch.motif_feat)
            else:
                logits = model(batch.x, batch.edge_index, batch.batch)
        probs = torch.softmax(logits.float(), dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        y_true.extend(batch.y.cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())
        y_prob.extend(probs.cpu().numpy().tolist())

    metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    return metrics, y_true, y_pred, y_prob


def train_gnn(
    train_graphs,
    val_graphs,
    test_graphs,
    input_dim: int,
    config: TrainingConfig = TrainingConfig(),
):
    if len(train_graphs) == 0 or len(val_graphs) == 0 or len(test_graphs) == 0:
        raise ValueError("train/val/test graph splits must be non-empty for production training.")

    if config.use_attention_fusion:
        if config.motif_dim <= 0:
            raise ValueError("use_attention_fusion=True requires motif_dim > 0")
        model = HybridGATWithAttention(
            input_dim=input_dim,
            motif_dim=config.motif_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        ).to(config.device)
    else:
        model = GATGraphClassifier(input_dim=input_dim, hidden_dim=config.hidden_dim, dropout=config.dropout).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    try:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=config.lr_scheduler_factor,
            patience=config.lr_scheduler_patience,
            verbose=False,
        )
    except TypeError:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=config.lr_scheduler_factor,
            patience=config.lr_scheduler_patience,
        )

    class_weights = _compute_class_weights(train_graphs, config.device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    if config.balance_datasets and any(hasattr(g, "dataset_id") for g in train_graphs):
        from collections import Counter
        ds_counts = Counter(getattr(g, "dataset_id", "unknown") for g in train_graphs)
        weights = [1.0 / ds_counts[getattr(g, "dataset_id", "unknown")] for g in train_graphs]
        sampler = WeightedRandomSampler(weights, num_samples=len(train_graphs), replacement=True)
        train_loader = DataLoader(train_graphs, batch_size=config.batch_size, sampler=sampler, num_workers=config.num_workers)
    else:
        train_loader = DataLoader(train_graphs, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_graphs, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_graphs, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    best_state = None
    best_val_auc = -1.0
    best_epoch = 0
    wait = 0

    history = []
    for epoch in range(1, config.max_epochs + 1):
        train_loss = _run_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            config.device,
            grad_clip_norm=config.grad_clip_norm,
            use_amp=config.use_amp,
        )
        val_metrics, _, _, _ = evaluate_gnn(model, val_loader, config.device, use_amp=config.use_amp)
        val_auc = val_metrics["roc_auc"]
        if not np.isfinite(val_auc):
            val_auc = val_metrics["balanced_accuracy"]
        scheduler.step(val_auc)

        current_lr = float(optimizer.param_groups[0]["lr"])

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "lr": current_lr,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
        )

        if val_auc > (best_val_auc + config.min_delta):
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            _save_checkpoint(model, config.checkpoint_path)
            wait = 0
        else:
            wait += 1

        if wait >= config.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        _load_checkpoint_if_exists(model, config.checkpoint_path)

    test_metrics, y_true, y_pred, y_prob = evaluate_gnn(model, test_loader, config.device, use_amp=config.use_amp)

    return {
        "model": model,
        "history": pd.DataFrame(history),
        "best_epoch": int(best_epoch),
        "best_val_auc": float(best_val_auc),
        "metrics": test_metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


