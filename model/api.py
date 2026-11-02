"""FastAPI backend. Loads saved weights + processed graph ONCE at startup (no retraining).

    uvicorn graphguard.api:app --reload --port 8000
Env: GG_ARTIFACTS (default artifacts), DATABASE_URL (default sqlite:///./notes.db)
"""
import json, os
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np, torch
from scipy.sparse import csr_matrix
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import BigInteger, Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from model.data import load_data
from model.models import build_model, predict_proba

ART = Path(os.getenv("GG_ARTIFACTS", "artifacts"))
engine = create_engine(os.getenv("DATABASE_URL", "sqlite:///./notes.db"), pool_pre_ping=True)
Session, Base = sessionmaker(engine), declarative_base()


class Note(Base):
    """Analyst review notes. These are NOT ground-truth labels."""
    __tablename__ = "notes"
    id = Column(Integer, primary_key=True)
    tx_id = Column(BigInteger, index=True)
    status = Column(String(32), default="open")
    note = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(engine)
app = FastAPI(title="GraphGuard")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

data = load_data(cache=ART / "processed.pt")
results = json.loads((ART / "results.json").read_text())
scores = {k: v for k, v in np.load(ART / "scores.npz").items()}

# Live inference from saved M2M-GNN weights (one forward pass, no training).
if (ART / "m2mgnn.pt").exists():
    ck = torch.load(ART / "m2mgnn.pt", map_location="cpu")
    net = build_model("m2mgnn", ck["in_dim"], ck["cfg"]); net.load_state_dict(ck["state_dict"])
    scores["m2mgnn"] = predict_proba(net, data.x, data.edge_index).astype(np.float32)

N = data.x.size(0)
TX = data.tx_id.numpy(); TIME = data.time.numpy(); Y = data.y.numpy()
ID2IDX = {int(t): i for i, t in enumerate(TX)}
SPLIT = np.full(N, "unlabeled", dtype=object)
SPLIT[data.train_mask.numpy()] = "train"; SPLIT[data.val_mask.numpy()] = "val"; SPLIT[data.test_mask.numpy()] = "test"
ed = data.edge_index_dir.numpy()
OUT = csr_matrix((np.ones(ed.shape[1]), (ed[0], ed[1])), shape=(N, N))
IN = OUT.T.tocsr(); UND = (OUT + IN).tocsr()
LABELS = {1: "illicit", 0: "licit", -1: "unknown"}


def thr(model): return results["models"][model]["threshold"]
def check_model(m):
    if m not in scores: raise HTTPException(404, f"unknown model {m}; have {list(scores)}")


def row(i, model):
    p = float(scores[model][i])
    return dict(tx_id=int(TX[i]), time_step=int(TIME[i]), label=LABELS[int(Y[i])], split=str(SPLIT[i]),
                score=round(p, 5), predicted_illicit=bool(p >= thr(model)))


@app.get("/api/health")
def health(): return dict(status="ok", models=list(scores), nodes=N)


@app.get("/api/metrics")
def metrics(): return results


@app.get("/api/transactions")
def transactions(model: str = "m2mgnn", split: str = "test", label: str = "all", min_score: float = 0.0,
                 predicted: Optional[bool] = None, sort: str = "score_desc",
                 limit: int = Query(50, le=500), offset: int = 0):
    check_model(model)
    p = scores[model]
    m = p >= min_score
    if split != "all": m &= SPLIT == split
    if label != "all": m &= Y == {v: k for k, v in LABELS.items()}[label]
    if predicted is not None: m &= (p >= thr(model)) == predicted
    idx = np.where(m)[0]
    idx = idx[np.argsort(-p[idx] if sort == "score_desc" else p[idx], kind="stable")]
    return dict(total=int(len(idx)), items=[row(i, model) for i in idx[offset:offset + limit]],
                threshold=thr(model), note="A score is a review-priority signal, not proof of wrongdoing.")


@app.get("/api/transactions/{tx_id}")
def transaction(tx_id: int, model: str = "m2mgnn"):
    check_model(model)
    if tx_id not in ID2IDX: raise HTTPException(404, "transaction not found")
    i = ID2IDX[tx_id]
    z = data.x[i].numpy(); top = np.argsort(-np.abs(z))[:8]
    nb = lambda M: [row(j, model) for j in M[i].indices[:25]]
    return dict(**row(i, model), all_scores={m: round(float(s[i]), 5) for m, s in scores.items()},
                top_features_standardized={f"f{int(k)}": round(float(z[k]), 3) for k in top},
                upstream=nb(IN), downstream=nb(OUT))


@app.get("/api/graph/{tx_id}")
def graph(tx_id: int, hops: int = Query(1, ge=1, le=3), model: str = "m2mgnn", max_nodes: int = 150):
    check_model(model)
    if tx_id not in ID2IDX: raise HTTPException(404, "transaction not found")
    root = ID2IDX[tx_id]; seen = {root: 0}; q = deque([root])
    while q and len(seen) < max_nodes:
        u = q.popleft()
        if seen[u] >= hops: continue
        for v in UND[u].indices:
            if v not in seen and len(seen) < max_nodes:
                seen[v] = seen[u] + 1; q.append(v)
    ids = list(seen); sub = OUT[ids][:, ids].tocoo()
    return dict(nodes=[row(i, model) | {"hop": seen[i]} for i in ids],
                edges=[dict(source=int(TX[ids[a]]), target=int(TX[ids[b]])) for a, b in zip(sub.row, sub.col)],
                truncated=len(seen) >= max_nodes)


class NoteIn(BaseModel):
    tx_id: int
    status: str = "open"
    note: str = ""


@app.post("/api/notes")
def add_note(n: NoteIn):
    if n.tx_id not in ID2IDX: raise HTTPException(404, "transaction not found")
    with Session() as s:
        o = Note(**n.model_dump()); s.add(o); s.commit(); s.refresh(o)
        return dict(id=o.id, tx_id=o.tx_id, status=o.status, note=o.note, created_at=o.created_at.isoformat())


@app.get("/api/notes")
def list_notes(tx_id: Optional[int] = None):
    with Session() as s:
        q = s.query(Note).order_by(Note.id.desc())
        if tx_id is not None: q = q.filter(Note.tx_id == tx_id)
        return [dict(id=o.id, tx_id=o.tx_id, status=o.status, note=o.note, created_at=o.created_at.isoformat())
                for o in q.limit(200)]
