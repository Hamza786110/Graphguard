"""GCN, GraphSAGE and M2M-GNN (written from Eqs. 7-10 of Liang et al., ICML 2024).

M2M-GNN here is NOT copied from the official repo: diff it against
github.com/Jinx-byebye/m2mgnn/m2mgnn/model.py and document deviations.
Edge convention (PyG): edge_index[0] = source j, edge_index[1] = target i.
"""
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv


class GCN(nn.Module):
    def __init__(self, i, h, o, layers=2, dropout=0.3, conv=GCNConv):
        super().__init__()
        dims = [i] + [h] * (layers - 1) + [o]
        self.convs = nn.ModuleList(conv(a, b) for a, b in zip(dims, dims[1:]))
        self.dropout = dropout

    def forward(self, x, ei):
        for k, c in enumerate(self.convs):
            x = c(F.dropout(x, self.dropout, self.training), ei)
            if k < len(self.convs) - 1:
                x = F.relu(x)
        return x


class SAGE(GCN):
    def __init__(self, i, h, o, layers=2, dropout=0.3):
        super().__init__(i, h, o, layers, dropout, conv=SAGEConv)


class M2MLayer(nn.Module):
    def __init__(self, d, C, alpha=0.5, tau=1.0, beta=0.5):
        super().__init__()
        assert d % C == 0
        self.d, self.C, self.dc = d, C, d // C
        self.alpha, self.tau, self.beta = alpha, tau, beta
        self.W = nn.Linear(d, self.dc, bias=False)
        self.W_att = nn.Linear(self.dc, C, bias=False)
        self.norm = nn.LayerNorm(d)
        self.last_scores = None

    def forward(self, h, h0, ei):
        n = h.size(0)
        src, dst = ei
        hh = self.W(h)
        s = F.softmax(self.W_att(F.relu(self.alpha * hh[dst] + hh[src])) / self.tau, dim=-1)  # Eq.7
        self.last_scores = s.detach()
        msg = s.unsqueeze(-1) * hh[src].unsqueeze(1)                                           # Eq.8
        m = hh.new_zeros(n, self.C, self.dc).index_add_(0, dst, msg).reshape(n, self.d)
        out = self.norm(F.relu((1 - self.beta) * h0 + self.beta * m))                          # Eq.9
        reg = (self.C ** 0.5) / s.size(0) * s.sum(0).norm(p=2) - 1.0                           # Eq.10 (verify)
        return out, reg


class M2MNet(nn.Module):
    def __init__(self, i, h, o, chunks=2, layers=2, dropout=0.3, alpha=0.5, tau=1.0, beta=0.5):
        super().__init__()
        self.enc1, self.enc2 = nn.Linear(i, h), nn.Linear(h, h)
        self.layers = nn.ModuleList(M2MLayer(h, chunks, alpha, tau, beta) for _ in range(layers))
        self.cls = nn.Linear(h, o)
        self.dropout = dropout

    def forward(self, x, ei):
        x = F.dropout(x, self.dropout, self.training)
        h0 = self.enc2(F.dropout(F.relu(self.enc1(x)), self.dropout, self.training))
        h, regs = h0, []
        for l in self.layers:
            h, r = l(h, h0, ei)
            h = F.dropout(h, self.dropout, self.training)
            regs.append(r)
        return self.cls(h), torch.stack(regs).mean()


def build_model(name, in_dim, cfg):
    if name == "gcn":
        return GCN(in_dim, cfg["hidden"], 2, cfg["layers"], cfg["dropout"])
    if name == "sage":
        return SAGE(in_dim, cfg["hidden"], 2, cfg["layers"], cfg["dropout"])
    if name == "m2mgnn":
        return M2MNet(in_dim, cfg["hidden"], 2, cfg["chunks"], cfg["layers"], cfg["dropout"],
                      cfg["alpha"], cfg["tau"], cfg["beta"])
    raise ValueError(name)


@torch.no_grad()
def predict_proba(model, x, ei):
    model.eval()
    out = model(x, ei)
    logits = out[0] if isinstance(out, tuple) else out
    return F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
