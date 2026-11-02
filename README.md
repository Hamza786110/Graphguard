# GraphGuard — M2M-GNN on the Elliptic Bitcoin dataset

Research prototype. A score is a prioritisation signal for human review, **not proof of wrongdoing**.
**Status: written but not yet executed** — run the steps below and fix whatever breaks; report only measured numbers.

## 1. Setup
```
pip install -r requirements.txt        # install torch first from pytorch.org if you want GPU
```
Download the Kaggle Elliptic dataset and put the three CSVs in `data/raw/`
(`elliptic_txs_features.csv`, `elliptic_txs_classes.csv`, `elliptic_txs_edgelist.csv`).

## 2. Train + evaluate (do this before anything else)
```
python train.py --models xgboost gcn sage m2mgnn --seeds 3
```
Start with `--seeds 1 --epochs 50` as a pilot on the 8 GB machine. Outputs in `artifacts/`:
`results.json` (precision/recall/F1/PR-AUC, FP/FN, per-time-step, timing), `scores.npz`, `m2mgnn.pt`, `processed.pt`.
Check `meta.cross_time_edges` in `results.json` is 0 (confirms no cross-time-step leakage).

## 3. Serve
```
uvicorn graphguard.api:app --port 8000        # sqlite notes locally
cd frontend && npm install && npm run dev      # http://localhost:5173
# or: docker compose up --build   (API + PostgreSQL; run the frontend with npm)
```
API: `/api/metrics`, `/api/transactions`, `/api/transactions/{id}`, `/api/graph/{id}?hops=`, `/api/notes`.

## Method notes / deviations to document
- Split: train steps 1–29, val 30–34, test 35–49; threshold tuned on val only; unknown nodes excluded from loss and metrics.
- `time_step` is not an input feature; features standardised with train statistics.
- GNNs use an undirected, self-loop-free graph (the paper assumes undirected). 2 chunks (illicit/licit).
- `graphguard/models.py::M2MLayer` is written from the paper (Eqs. 7–10), not copied from the official repo — diff it against
  `github.com/Jinx-byebye/m2mgnn` and record deviations (esp. Eq. 10, dropout placement).
- Class imbalance handled with a weighted cross-entropy (XGBoost: `scale_pos_weight`).
- Not implemented from the proposal: Plotly charts (metrics are tables), account-level detection.
