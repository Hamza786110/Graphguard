"""Train + evaluate XGBoost, GCN, GraphSAGE and M2M-GNN on Elliptic.

    python train.py --models xgboost gcn sage m2mgnn --seeds 3

Writes artifacts/results.json, artifacts/scores.npz, artifacts/<model>.pt|json.
Every number in results.json is measured here; nothing is hard-coded.
"""
import argparse, json, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from sklearn.metrics import average_precision_score, precision_recall_curve
from graphguard_.data import load_data, TRAIN_END, VAL_END, LAST_STEP
from graphguard_.models import build_model, predict_proba

CFG = dict(hidden=64, layers=2, dropout=0.3, lr=0.005, wd=5e-4, epochs=300, patience=12,
           lam=0.5, chunks=2, alpha=0.5, tau=1.0, beta=0.5)
KEYS = ["precision", "recall", "f1", "pr_auc"]


def metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(precision=pr, recall=rc, f1=2 * pr * rc / max(pr + rc, 1e-12),
                pr_auc=float(average_precision_score(y, p)) if y.sum() > 0 else None,
                tp=tp, fp=fp, fn=fn, n=int(len(y)), n_illicit=int(y.sum()))


def best_threshold(y, p):
    pr, rc, thr = precision_recall_curve(y, p)
    f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-12)
    return float(thr[int(np.argmax(f1[:-1]))])


def finish(data, probs, secs, extra):
    y, t = data.y.numpy(), data.time.numpy()
    val, test = data.val_mask.numpy(), data.test_mask.numpy()
    thr = best_threshold(y[val], probs[val])           # threshold chosen on validation only
    steps = []
    for s in range(VAL_END + 1, LAST_STEP + 1):
        m = test & (t == s)
        steps.append(dict(time_step=s, **metrics(y[m], probs[m], thr)) if m.any() else dict(time_step=s))
    return dict(threshold=thr, val=metrics(y[val], probs[val], thr),
                test=metrics(y[test], probs[test], thr), per_timestep=steps,
                train_seconds=round(secs, 1), **extra)


def rss_mb():
    try:
        import psutil, os
        return round(psutil.Process(os.getpid()).memory_info().rss / 2**20, 1)
    except Exception:
        return None


def run_xgb(data, cfg, seed):
    from xgboost import XGBClassifier
    x, y = data.x.numpy(), data.y.numpy()
    tr, va = data.train_mask.numpy(), data.val_mask.numpy()
    clf = XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.08, subsample=0.8,
                        colsample_bytree=0.8, tree_method="hist", eval_metric="aucpr",
                        scale_pos_weight=float((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)),
                        early_stopping_rounds=30, random_state=seed, n_jobs=-1)
    t0 = time.time()
    clf.fit(x[tr], y[tr], eval_set=[(x[va], y[va])], verbose=False)
    secs = time.time() - t0
    return clf.predict_proba(x)[:, 1], secs, clf


def run_gnn(name, data, cfg, seed, device):
    torch.manual_seed(seed); np.random.seed(seed)
    model = build_model(name, data.x.size(1), cfg).to(device)
    x, ei, y = data.x.to(device), data.edge_index.to(device), data.y.to(device)
    tr = data.train_mask.to(device)
    va = data.val_mask.numpy(); yv = data.y.numpy()[va]
    w = torch.tensor([1.0, float((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1))], device=device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    best_ap, best_state, bad = -1, None, 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for ep in range(1, cfg["epochs"] + 1):
        model.train(); opt.zero_grad()
        out = model(x, ei)
        logits, reg = out if isinstance(out, tuple) else (out, 0.0)
        loss = F.cross_entropy(logits[tr], y[tr], weight=w) + (cfg["lam"] * reg if name == "m2mgnn" else 0.0)
        loss.backward(); opt.step()
        if ep % 5 == 0:
            ap = average_precision_score(yv, predict_proba(model, x, ei)[va])
            if ap > best_ap:
                best_ap, bad = ap, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg["patience"]:
                    break
    secs = time.time() - t0
    model.load_state_dict(best_state)
    probs = predict_proba(model, x, ei)
    extra = dict(best_epoch_val_pr_auc=float(best_ap), epochs_run=ep)
    if device.type == "cuda":
        extra["peak_gpu_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    extra["rss_mb_after_training"] = rss_mb()
    if name == "m2mgnn":
        s = model.layers[0].last_scores
        extra["chunk_usage_layer1"] = (torch.bincount(s.argmax(-1), minlength=s.size(1)).float() / s.size(0)).tolist()
    return probs, secs, (model, extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw"); ap.add_argument("--out", default="artifacts")
    ap.add_argument("--models", nargs="+", default=["xgboost", "gcn", "sage", "m2mgnn"])
    ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args()
    cfg = dict(CFG); cfg["epochs"] = a.epochs or cfg["epochs"]
    out = Path(a.out); out.mkdir(exist_ok=True)
    data = load_data(a.raw_dir, out / "processed.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("data:", data.meta, "| device:", device)
    results, scores = dict(meta=data.meta, config=cfg, split=dict(train=f"1-{TRAIN_END}",
                           val=f"{TRAIN_END+1}-{VAL_END}", test=f"{VAL_END+1}-{LAST_STEP}"), models={}), {}
    for name in a.models:
        runs, best = [], None
        for seed in range(a.seeds):
            if name == "xgboost":
                probs, secs, obj = run_xgb(data, cfg, seed); extra = dict(rss_mb_after_training=rss_mb())
            else:
                probs, secs, (obj, extra) = run_gnn(name, data, cfg, seed, device)
            r = finish(data, probs, secs, extra); r["seed"] = seed; runs.append(r)
            print(f"{name} seed{seed} test", {k: round(r['test'][k], 4) for k in KEYS if r['test'][k] is not None})
            if best is None or r["val"]["pr_auc"] > best[0]["val"]["pr_auc"]:
                best = (r, probs, obj)
        agg = {k: dict(mean=float(np.mean([r["test"][k] for r in runs])),
                       std=float(np.std([r["test"][k] for r in runs]))) for k in KEYS}
        results["models"][name] = dict(runs=runs, test_mean_std=agg, best_run=best[0],
                                       threshold=best[0]["threshold"])
        scores[name] = best[1].astype(np.float32)
        if name == "xgboost":
            best[2].save_model(str(out / "xgboost.json"))
        else:
            torch.save(dict(state_dict=best[2].state_dict(), cfg=cfg, in_dim=int(data.x.size(1)),
                            threshold=best[0]["threshold"]), out / f"{name}.pt")
    np.savez(out / "scores.npz", **scores)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print("saved ->", out)


if __name__ == "__main__":
    main()
