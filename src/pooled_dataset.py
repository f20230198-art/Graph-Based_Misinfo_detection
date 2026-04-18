from __future__ import annotations

from typing import List, Sequence, Tuple

from sklearn.model_selection import train_test_split
from torch_geometric.data import Data


def concat_datasets(*dataset_lists: Sequence[Data]) -> List[Data]:
    pooled: List[Data] = []
    for ds in dataset_lists:
        pooled.extend(ds)
    return pooled


def _stratum_key(g: Data) -> str:
    label = int(g.y.item())
    ds = getattr(g, "dataset_id", "unknown")
    return f"{ds}|{label}"


def stratified_pooled_split(
    graphs: Sequence[Data],
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[List[Data], List[Data], List[Data]]:
    indices = list(range(len(graphs)))
    strata = [_stratum_key(g) for g in graphs]

    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=strata,
    )

    relative_val_size = val_size / (1.0 - test_size)
    train_strata = [strata[i] for i in train_idx]
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=train_strata,
    )

    train = [graphs[i] for i in train_idx]
    val = [graphs[i] for i in val_idx]
    test = [graphs[i] for i in test_idx]
    return train, val, test


def split_by_dataset(graphs: Sequence[Data]) -> dict:
    buckets: dict = {}
    for g in graphs:
        ds = getattr(g, "dataset_id", "unknown")
        buckets.setdefault(ds, []).append(g)
    return buckets
