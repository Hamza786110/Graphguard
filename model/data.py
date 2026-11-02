"""
Elliptic Bitcoin dataset -> PyG-style tensors with a leak-free temporal split.

Raw files (Kaggle "ellipticco/elliptic-data-set") go in data/raw/:
  elliptic_txs_features.csv  (no header: txId, time_step, 165 features)
  elliptic_txs_classes.csv   (txId, class in {"1" illicit, "2" licit, "unknown"})
  elliptic_txs_edgelist.csv  (txId1, txId2)  directed flow-of-funds edges

Design decisions (document these in the report):
  * labels: illicit=1, licit=0, unknown=-1. Unknown nodes stay in the graph for
    message passing but are NEVER in any loss / metric.
  * time_step is NOT used as an input feature (it would not generalise to later steps).
  * split by time: train 1-29, val 30-34, test 35-49 (labelled nodes only).
  * feature standardisation uses train-node statistics only.
  * GNNs use an undirected copy of the graph without self-loops (the M2M paper assumes
    undirected graphs); the directed edges are kept for the case/graph views.
  * Elliptic edges are expected to connect transactions inside the same time step, so a
    full-graph forward pass should not leak future information into earlier steps.
    meta["cross_time_edges"] counts violations -- verify it is 0 on your files.
"""
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, remove_self_loops

TRAIN_END, VAL_END, LAST_STEP = 29, 34, 49


def build_elliptic(raw_dir):
    raw = Path(raw_dir)
    feats = pd.read_csv(raw / "elliptic_txs_features.csv", header=None)
    classes = pd.read_csv(raw / "elliptic_txs_classes.csv")
    edges = pd.read_csv(raw / "elliptic_txs_edgelist.csv")

    tx_ids = feats[0].to_numpy(np.int64)
    time = feats[1].to_numpy(np.int64)
    x = feats.iloc[:, 2:].to_numpy(np.float32)
    n = len(tx_ids)

    lab = classes.set_index("txId")["class"].astype(str).reindex(tx_ids)
    y = lab.map({"1": 1, "2": 0}).fillna(-1).to_numpy(np.int64)

    idx = pd.Series(np.arange(n), index=tx_ids)
    s = idx.reindex(edges.iloc[:, 0].to_numpy()).to_numpy()
    t = idx.reindex(edges.iloc[:, 1].to_numpy()).to_numpy()
    ok = ~(np.isnan(s) | np.isnan(t))
    s, t = s[ok].astype(np.int64), t[ok].astype(np.int64)
    cross_time_edges = int((time[s] != time[t]).sum())
    ei_dir, _ = remove_self_loops(torch.from_numpy(np.stack([s, t])))
    ei = to_undirected(ei_dir, num_nodes=n)

    labeled = y >= 0
    train = labeled & (time <= TRAIN_END)
    val = labeled & (time > TRAIN_END) & (time <= VAL_END)
    test = labeled & (time > VAL_END)
    mu, sd = x[train].mean(0), x[train].std(0) + 1e-6
    x = (x - mu) / sd

    return dict(
        x=torch.from_numpy(x), y=torch.from_numpy(y), time=torch.from_numpy(time),
        tx_id=torch.from_numpy(tx_ids), edge_index=ei, edge_index_dir=ei_dir,
        train_mask=torch.from_numpy(train), val_mask=torch.from_numpy(val),
        test_mask=torch.from_numpy(test),
        meta=dict(num_nodes=n, num_edges_directed=int(ei_dir.size(1)),
                  num_features=int(x.shape[1]), cross_time_edges=cross_time_edges,
                  n_illicit=int((y == 1).sum()), n_licit=int((y == 0).sum()),
                  n_unknown=int((y == -1).sum()),
                  train_nodes=int(train.sum()), val_nodes=int(val.sum()), test_nodes=int(test.sum())),
    )


def load_data(raw_dir="data/raw", cache="artifacts/processed.pt"):
    cache = Path(cache)
    if cache.exists():
        d = torch.load(cache, weights_only=False)
    else:
        d = build_elliptic(raw_dir)
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(d, cache)
    meta = d.pop("meta")
    data = Data(**d)
    data.meta = meta
    return data
