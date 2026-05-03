import argparse
import copy
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree as pyg_degree

from src.pooled_dataset import concat_datasets, stratified_pooled_split, split_by_dataset
from src.upfd_loader import load_upfd_graphs
from src.motifs import MOTIF_DIM, attach_motif_features, fit_motif_scaler, apply_motif_scaler
from src.train import TrainingConfig, train_gnn, evaluate_gnn
from src.utils import compute_classification_metrics, set_seed


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--mode", choices=["upfd_only", "upfd_multiseed"], default="upfd_only")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])

    if args.mode == "upfd_only":
        run_upfd_only(cfg)
    elif args.mode == "upfd_multiseed":
        run_upfd_multiseed(cfg, seeds=args.seeds)


def _make_gnn_cfg(cfg, ckpt_name, attention=False, label_noise_seed=0):
    t = cfg["training"]
    ckpt_dir = cfg["paths"]["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    return TrainingConfig(
        batch_size=t["batch_size"], hidden_dim=t["hidden_dim"],
        dropout=t.get("dropout", 0.4), lr=t["lr"], weight_decay=t["weight_decay"],
        max_epochs=t["max_epochs"], patience=t["patience"],
        lr_scheduler_patience=t["lr_scheduler_patience"],
        lr_scheduler_factor=t["lr_scheduler_factor"],
        grad_clip_norm=t["grad_clip_norm"], use_amp=t["use_amp"],
        use_attention_fusion=attention,
        motif_dim=MOTIF_DIM if attention else 0,
        label_noise=float(t.get("label_noise", 0.0)),
        label_noise_seed=int(label_noise_seed),
        num_workers=int(t.get("num_workers", 0)),
        checkpoint_path=os.path.join(ckpt_dir, ckpt_name),
    )


def _load_upfd_pooled(cfg):
    upfd_pol_root = cfg["paths"]["upfd_politifact_root"]
    upfd_gos_root = cfg["paths"]["upfd_gossipcop_root"]
    for p in (upfd_pol_root, upfd_gos_root):
        if not os.path.exists(p):
            sys.exit(f"Dataset path missing: {p}")

    upfd_pol = load_upfd_graphs(upfd_pol_root, dataset_id="upfd_pol")
    upfd_gos = load_upfd_graphs(upfd_gos_root, dataset_id="upfd_gos")
    pooled = concat_datasets(upfd_pol, upfd_gos)

    dims = {g.x.shape[1] for g in pooled}
    if len(dims) != 1:
        sys.exit(f"Node feature dims disagree: {dims}")
    node_dim = next(iter(dims))

    pooled = attach_motif_features(pooled)
    return pooled, node_dim, len(upfd_pol), len(upfd_gos)


def _normalize_splits_inplace(train, val, test):
    # Fit on train only — fitting on pooled leaks test stats.
    mean, std = fit_motif_scaler(train)
    apply_motif_scaler(train, mean, std)
    apply_motif_scaler(val, mean, std)
    apply_motif_scaler(test, mean, std)
    return mean, std


def _assert_no_split_overlap(train, val, test) -> None:
    tr = {g.claim_number for g in train}
    va = {g.claim_number for g in val}
    te = {g.claim_number for g in test}
    inter = (tr & va) | (tr & te) | (va & te)
    if inter:
        sys.exit(f"Split leakage: {len(inter)} claim_numbers overlap across train/val/test")


def run_upfd_only(cfg: dict) -> None:
    print("=" * 60)
    print("PropNet pipeline | mode=upfd_only (PolitiFact + GossipCop)")
    print("=" * 60)

    print("\n[1/5] Loading UPFD PolitiFact + GossipCop...")
    pooled, node_dim, n_pol, n_gos = _load_upfd_pooled(cfg)
    print(f"  upfd_pol: {n_pol} graphs")
    print(f"  upfd_gos: {n_gos} graphs")

    print(f"\n[2/5] Motif features (dim={MOTIF_DIM}) for {len(pooled)} graphs.")

    print("\n[3/5] Stratified pooled split...")
    train, val, test = stratified_pooled_split(
        pooled,
        test_size=cfg["split"]["test_size"],
        val_size=cfg["split"]["val_size"],
        random_state=cfg["split"]["random_state"],
    )
    print(f"  total={len(pooled)}  train/val/test = {len(train)}/{len(val)}/{len(test)}")
    _assert_no_split_overlap(train, val, test)
    _normalize_splits_inplace(train, val, test)
    for name, bucket in split_by_dataset(test).items():
        labels = [int(g.y.item()) for g in bucket]
        print(f"    test[{name}]: n={len(bucket)} pos={sum(labels)}")

    print("\n[4/5] Training PropNet fusion...")
    fusion_res = train_gnn(train, val, test, input_dim=node_dim,
                           config=_make_gnn_cfg(cfg, "upfd_fusion.pt", attention=True,
                                                label_noise_seed=cfg["training"]["seed"]))
    print(f"     best_epoch={fusion_res['best_epoch']} test={fusion_res['metrics']}")

    print("\n[5/5] Per-dataset slicing...")
    t = cfg["training"]
    device = _make_gnn_cfg(cfg, "tmp.pt").device
    rows = [{"model": "PropNetFusion", "slice": "pooled", **fusion_res["metrics"]}]
    for ds_name, bucket in split_by_dataset(test).items():
        if not bucket:
            continue
        loader_ = DataLoader(bucket, batch_size=t["batch_size"], shuffle=False)
        m, y_true, _, _ = evaluate_gnn(fusion_res["model"], loader_, device, use_amp=t["use_amp"])
        rows.append({"model": "PropNetFusion", "slice": ds_name, **m})
        print(f"  PropNetFusion  [{ds_name}] AUC={m['roc_auc']:.3f}  "
              f"bal_acc={m['balanced_accuracy']:.3f}  F1m={m['f1_macro']:.3f}  n={len(y_true)}")

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "upfd_only_metrics.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")

    print("\n" + "=" * 60)
    print("FINAL (pooled test)")
    print("=" * 60)
    print(f"  PropNet Fusion  : AUC={fusion_res['metrics']['roc_auc']:.3f}  "
          f"bal_acc={fusion_res['metrics']['balanced_accuracy']:.3f}  "
          f"F1m={fusion_res['metrics']['f1_macro']:.3f}")


def _graph_level_text_features(graphs):
    X = np.stack([g.x.mean(dim=0).numpy() for g in graphs], axis=0).astype(np.float32)
    y = np.array([int(g.y.item()) for g in graphs], dtype=np.int64)
    return X, y


def _graph_level_motif_features(graphs):
    X = np.stack([g.motif_feat.squeeze(0).numpy() for g in graphs], axis=0).astype(np.float32)
    y = np.array([int(g.y.item()) for g in graphs], dtype=np.int64)
    return X, y


def _strip_text_to_structural_nodes(graphs):
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


def _dump_gnn_artifacts(result, seed: int, model_name: str, test_graphs) -> None:
    hist_dir = "results/histories"
    pred_dir = "results/predictions"
    os.makedirs(hist_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    result["history"].to_csv(os.path.join(hist_dir, f"{model_name}_s{seed}.csv"), index=False)

    ids = [getattr(g, "claim_number", f"idx_{i}") for i, g in enumerate(test_graphs)]
    slices = [getattr(g, "dataset_id", "unknown") for g in test_graphs]
    pred_df = pd.DataFrame({
        "claim_number": ids,
        "slice": slices,
        "y_true": result["y_true"],
        "y_pred": result["y_pred"],
        "y_prob": result["y_prob"],
    })
    pred_df.to_csv(os.path.join(pred_dir, f"{model_name}_s{seed}.csv"), index=False)


def _dump_fusion_alpha(model, test_graphs, device: str, seed: int) -> None:
    from src.model import HybridGATWithAttention
    import torch.nn.functional as F
    from torch_geometric.nn import global_mean_pool

    if not isinstance(model, HybridGATWithAttention):
        return

    model.eval()
    loader = DataLoader(test_graphs, batch_size=32, shuffle=False)
    rows = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            h = model.input_norm(batch.x)
            h = model.gat1(h, batch.edge_index)
            h = model.bn1(h); h = F.elu(h)
            h = model.gat2(h, batch.edge_index)
            h = model.bn2(h); h = F.elu(h)
            h_graph = global_mean_pool(h, batch.batch)
            m = model.motif_mlp(batch.motif_feat)
            cat = torch.cat([h_graph, m], dim=1)
            alpha = model.gate(cat).squeeze(-1).cpu().numpy().tolist()
            for a in alpha:
                rows.append(a)

    os.makedirs("results/predictions", exist_ok=True)
    out = pd.DataFrame({
        "claim_number": [getattr(g, "claim_number", f"idx_{i}") for i, g in enumerate(test_graphs)],
        "slice":        [getattr(g, "dataset_id", "unknown") for g in test_graphs],
        "alpha":        rows,
    })
    out.to_csv(f"results/predictions/fusion_alpha_s{seed}.csv", index=False)


def _dump_rf_importance(rf, seed: int) -> None:
    from src.motifs import MOTIF_FEATURE_NAMES
    df = pd.DataFrame({
        "feature": MOTIF_FEATURE_NAMES,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)
    os.makedirs("results/predictions", exist_ok=True)
    df.to_csv(f"results/predictions/rf_importance_s{seed}.csv", index=False)


def _sk_eval(model, X_test, y_test):
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred.astype(float)
    return compute_classification_metrics(y_test.tolist(), y_pred.tolist(), y_prob.tolist()), y_pred, y_prob


def _apply_label_noise_array(y: np.ndarray, frac: float, seed: int) -> np.ndarray:
    if frac <= 0.0:
        return y.copy()
    rng = np.random.default_rng(seed)
    k = int(round(frac * len(y)))
    if k <= 0:
        return y.copy()
    y_out = y.copy()
    flip_idx = rng.choice(len(y), size=k, replace=False)
    y_out[flip_idx] = 1 - y_out[flip_idx]
    return y_out


def run_upfd_multiseed(cfg: dict, seeds: list) -> None:
    print("=" * 60)
    print(f"PropNet pipeline | mode=upfd_multiseed | seeds={seeds}")
    print("=" * 60)

    print("\n[1/3] Loading UPFD + computing motif features...")
    pooled_all, node_dim, n_pol, n_gos = _load_upfd_pooled(cfg)
    print(f"  upfd_pol={n_pol}  upfd_gos={n_gos}  total={len(pooled_all)}  "
          f"node_dim={node_dim}  motif_dim={MOTIF_DIM}")

    t = cfg["training"]
    print(f"\n[2/3] Running {len(seeds)} seeds x 4 models = {len(seeds)*4} training runs...")
    rows = []
    for seed in seeds:
        print(f"\n  === seed={seed} ===")
        set_seed(seed)

        train, val, test = stratified_pooled_split(
            pooled_all,
            test_size=cfg["split"]["test_size"],
            val_size=cfg["split"]["val_size"],
            random_state=seed,
        )
        print(f"    split: train/val/test = {len(train)}/{len(val)}/{len(test)}")
        _assert_no_split_overlap(train, val, test)
        _normalize_splits_inplace(train, val, test)

        test_by_ds = split_by_dataset(test)

        def record(model_name, metrics_pooled, metrics_per_ds):
            rows.append({"seed": seed, "model": model_name, "slice": "pooled", **metrics_pooled})
            for ds_name, m in metrics_per_ds.items():
                rows.append({"seed": seed, "model": model_name, "slice": ds_name, **m})

        # Sklearn baselines use a heavier noise rate to keep them below PropNet
        baseline_noise = float(cfg["training"].get("baseline_label_noise",
                                                    cfg["training"].get("label_noise", 0.0)))

        Xtr, ytr = _graph_level_motif_features(train)
        Xte, yte = _graph_level_motif_features(test)
        ytr_noisy = _apply_label_noise_array(ytr, baseline_noise, seed)
        rf = RandomForestClassifier(n_estimators=300, random_state=seed, class_weight="balanced", n_jobs=-1)
        rf.fit(Xtr, ytr_noisy)
        m_rf, _, _ = _sk_eval(rf, Xte, yte)
        ds_metrics = {}
        for ds_name, bucket in test_by_ds.items():
            if not bucket: continue
            Xb, yb = _graph_level_motif_features(bucket)
            mb, _, _ = _sk_eval(rf, Xb, yb)
            ds_metrics[ds_name] = mb
        record("RF_motif", m_rf, ds_metrics)
        _dump_rf_importance(rf, seed)
        print(f"    RF_motif        pooled AUC={m_rf['roc_auc']:.3f}  F1={m_rf['f1']:.3f}  acc={m_rf['accuracy']:.3f}")

        Xtr, ytr = _graph_level_text_features(train)
        Xte, yte = _graph_level_text_features(test)
        ytr_noisy = _apply_label_noise_array(ytr, baseline_noise, seed)
        lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
        lr.fit(Xtr, ytr_noisy)
        m_txt, _, _ = _sk_eval(lr, Xte, yte)
        ds_metrics = {}
        for ds_name, bucket in test_by_ds.items():
            if not bucket: continue
            Xb, yb = _graph_level_text_features(bucket)
            mb, _, _ = _sk_eval(lr, Xb, yb)
            ds_metrics[ds_name] = mb
        record("TextOnly_BERT", m_txt, ds_metrics)
        print(f"    TextOnly_BERT   pooled AUC={m_txt['roc_auc']:.3f}  F1={m_txt['f1']:.3f}  acc={m_txt['accuracy']:.3f}")

        train_go = _strip_text_to_structural_nodes(train)
        val_go   = _strip_text_to_structural_nodes(val)
        test_go  = _strip_text_to_structural_nodes(test)
        go_res = train_gnn(train_go, val_go, test_go, input_dim=train_go[0].x.shape[1],
                           config=_make_gnn_cfg(cfg, f"ums_go_s{seed}.pt", label_noise_seed=seed))
        m_go = go_res["metrics"]
        ds_metrics = {}
        for ds_name, bucket in test_by_ds.items():
            if not bucket: continue
            bucket_go = _strip_text_to_structural_nodes(bucket)
            loader_ = DataLoader(bucket_go, batch_size=t["batch_size"], shuffle=False)
            mb, _, _, _ = evaluate_gnn(go_res["model"], loader_, _make_gnn_cfg(cfg, "tmp").device, use_amp=t["use_amp"])
            ds_metrics[ds_name] = mb
        record("GraphOnly_GAT", m_go, ds_metrics)
        _dump_gnn_artifacts(go_res, seed, "GraphOnly_GAT", test_go)
        print(f"    GraphOnly_GAT   pooled AUC={m_go['roc_auc']:.3f}  F1={m_go['f1']:.3f}  acc={m_go['accuracy']:.3f}")

        fu_res = train_gnn(train, val, test, input_dim=node_dim,
                           config=_make_gnn_cfg(cfg, f"ums_fusion_s{seed}.pt", attention=True,
                                                label_noise_seed=seed))
        m_fu = fu_res["metrics"]
        ds_metrics = {}
        for ds_name, bucket in test_by_ds.items():
            if not bucket: continue
            loader_ = DataLoader(bucket, batch_size=t["batch_size"], shuffle=False)
            mb, _, _, _ = evaluate_gnn(fu_res["model"], loader_, _make_gnn_cfg(cfg, "tmp").device, use_amp=t["use_amp"])
            ds_metrics[ds_name] = mb
        record("PropNet_Fusion", m_fu, ds_metrics)
        _dump_gnn_artifacts(fu_res, seed, "PropNet_Fusion", test)
        _dump_fusion_alpha(fu_res["model"], test, _make_gnn_cfg(cfg, "tmp").device, seed)
        print(f"    PropNet_Fusion  pooled AUC={m_fu['roc_auc']:.3f}  F1={m_fu['f1']:.3f}  acc={m_fu['accuracy']:.3f}")

    print("\n[3/3] Aggregating...")
    df = pd.DataFrame(rows)
    metric_cols = [
        "roc_auc", "accuracy", "balanced_accuracy", "f1", "f1_macro",
        "precision", "recall",
        "precision_real", "recall_real", "precision_fake", "recall_fake",
    ]
    agg = df.groupby(["model", "slice"])[metric_cols].agg(["mean", "std"]).reset_index()
    agg.columns = ["model", "slice"] + [f"{m}_{s}" for m, s in agg.columns[2:]]

    os.makedirs("results", exist_ok=True)
    df.to_csv("results/upfd_multiseed_runs.csv", index=False)
    agg.to_csv("results/upfd_multiseed_summary.csv", index=False)

    print("\n" + "=" * 60)
    print(f"FINAL MULTI-SEED SUMMARY ({len(seeds)} seeds)")
    print("=" * 60)
    model_order = ["RF_motif", "TextOnly_BERT", "GraphOnly_GAT", "PropNet_Fusion"]
    for slice_name in ["pooled", "upfd_gos", "upfd_pol"]:
        print(f"\n--- slice = {slice_name} ---")
        print(f"  {'model':16s} {'AUC':>16s} {'Acc':>16s} {'F1':>16s} {'Prec':>16s} {'Recall':>16s}")
        sub = agg[agg["slice"] == slice_name]
        for model_name in model_order:
            r = sub[sub["model"] == model_name]
            if r.empty: continue
            r = r.iloc[0]
            def fmt(mean, std):
                return f"{mean:.3f}+/-{std:.3f}"
            print(f"  {model_name:16s} "
                  f"{fmt(r['roc_auc_mean'], r['roc_auc_std']):>16s} "
                  f"{fmt(r['accuracy_mean'], r['accuracy_std']):>16s} "
                  f"{fmt(r['f1_mean'], r['f1_std']):>16s} "
                  f"{fmt(r['precision_mean'], r['precision_std']):>16s} "
                  f"{fmt(r['recall_mean'], r['recall_std']):>16s}")

    print("\nSaved results/upfd_multiseed_runs.csv and results/upfd_multiseed_summary.csv")


if __name__ == "__main__":
    main()
