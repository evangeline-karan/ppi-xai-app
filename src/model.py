"""Phase 2 model, node features and explanations. Needs torch and torch_geometric (live mode only).

The feature layout and the model are identical to the code used for training in Colab
(see training/phase2_full_setup_and_train.py), so the saved weights can be loaded directly.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.utils import softmax as pyg_softmax

AA = "ACDEFGHIKLMNPQRSTVWY"
aa2i = {a: i for i, a in enumerate(AA)}
_hydro = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4, H=-3.2, I=4.5,
              L=3.8, K=-3.9, M=1.9, F=2.8, P=-1.6, S=-0.8, T=-0.7, W=-0.9, Y=-1.3, V=4.2)
_vol = dict(A=88.6, R=173.4, N=114.1, D=111.1, C=108.5, Q=143.8, E=138.4, G=60.1, H=153.2, I=166.7,
            L=166.7, K=168.6, M=162.9, F=189.9, P=112.7, S=89.0, T=116.1, W=227.8, Y=193.6, V=140.0)
_charge = {a: 0.0 for a in AA}
_charge.update(K=1, R=1, D=-1, E=-1, H=0.1)
PHYS = np.array([[_hydro[a] / 4.5, (_vol[a] - 130) / 50, _charge[a],
                  float(a in "FWY"), float(a in "STNQ")] for a in AA], dtype=np.float32)

MUT_FLAG = 1305          # column of the "this residue is mutated" flag in the node features
CUTOFF = 8.0             # C-alpha distance for edges inside one side (A)
PAIR_CUT = 15.0          # ligand-receptor pairs closer than this form the interface pair matrix (A)
PAIR_MAX = 3000          # at most this many pairs per complex (nearest first)
CONTACT = 8.0            # contact definition used for the auxiliary task and for "interface residue"
FOLLOW = ["x_l", "x_r", "pair_l"]


class PairData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key in ("edge_index_l", "pair_l"):
            return self.x_l.size(0)
        if key in ("edge_index_r", "pair_r"):
            return self.x_r.size(0)
        return super().__inc__(key, value, *args, **kwargs)


class NodeEncoder(nn.Module):
    def __init__(self, in_dim=1351, hid=256, heads=4, layers=3, drop=0.2):
        super().__init__()
        self.drop = drop
        self.inp = nn.Sequential(nn.Linear(in_dim, hid), nn.LayerNorm(hid), nn.ReLU(), nn.Dropout(drop))
        self.convs = nn.ModuleList([GATv2Conv(hid, hid // heads, heads=heads, dropout=drop) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hid) for _ in range(layers)])

    def forward(self, x, edge_index):
        h = self.inp(x)
        for conv, norm in zip(self.convs, self.norms):
            h = norm(h + F.dropout(F.relu(conv(h, edge_index)), self.drop, self.training))
        return h


class Phase2(nn.Module):
    def __init__(self, in_dim=1351, hid=256, pd=128, drop=0.2):
        super().__init__()
        self.enc_l, self.enc_r = NodeEncoder(in_dim, hid, drop=drop), NodeEncoder(in_dim, hid, drop=drop)
        self.proj_l, self.proj_r = nn.Linear(hid, pd), nn.Linear(hid, pd)
        self.pair_mlp = nn.Sequential(nn.Linear(2 * pd + 2, pd), nn.ReLU(), nn.Dropout(drop),
                                      nn.Linear(pd, pd), nn.ReLU())
        self.contact_head, self.attn_head = nn.Linear(pd, 1), nn.Linear(pd, 1)
        self.head = nn.Sequential(nn.Linear(2 * hid + pd, 256), nn.ReLU(), nn.Dropout(drop),
                                  nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, b, x_l=None, x_r=None):
        x_l = b.x_l if x_l is None else x_l
        x_r = b.x_r if x_r is None else x_r
        hl, hr = self.enc_l(x_l, b.edge_index_l), self.enc_r(x_r, b.edge_index_r)
        a, c = self.proj_l(hl)[b.pair_l], self.proj_r(hr)[b.pair_r]
        mut = torch.stack([x_l[b.pair_l, MUT_FLAG], x_r[b.pair_r, MUT_FLAG]], dim=1)
        z = self.pair_mlp(torch.cat([a * c, (a - c).abs(), mut], dim=1))
        contact_logit = self.contact_head(z).squeeze(-1)
        pb = b.pair_l_batch
        w = pyg_softmax(self.attn_head(z).squeeze(-1), pb)
        g = torch.zeros(b.num_graphs, z.size(1), device=z.device).index_add_(0, pb, w.unsqueeze(-1) * z)
        pooled = torch.cat([global_mean_pool(hl, b.x_l_batch), global_mean_pool(hr, b.x_r_batch), g], dim=1)
        return self.head(pooled).squeeze(-1), contact_logit, w


def load_model(path, device="cpu"):
    """Load phase2_weights.pt. Returns the model and the numbers needed to turn outputs into pKd."""
    ckpt = torch.load(path, map_location=device)
    model = Phase2().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    meta = {k: float(ckpt[k]) for k in ("y_mean", "y_std", "slope", "intercept")}
    return model, meta


def build_side(seq, emb, ca, muts):
    """Node features and edges for one side. muts = [(node_index, wild_type, mutant)] within this side."""
    idx = np.array([aa2i[a] for a in seq])
    onehot = np.eye(20, dtype=np.float32)[idx]
    blk = np.zeros((len(seq), 46), np.float32)
    for n, wt, mt in muts:
        blk[n, 0] = 1
        blk[n, 1 + aa2i[wt]] = 1
        blk[n, 21 + aa2i[mt]] = 1
        blk[n, 41:46] = PHYS[aa2i[mt]] - PHYS[aa2i[wt]]
    x = np.concatenate([np.asarray(emb, np.float32), onehot, PHYS[idx], blk], axis=1)
    c = torch.from_numpy(np.asarray(ca, np.float32))
    d = torch.cdist(c, c)
    mask = (d < CUTOFF) & ~torch.eye(len(seq), dtype=torch.bool)
    return torch.from_numpy(x), mask.nonzero().t().contiguous().long()


def interface_pairs(ca_l, ca_r):
    d = torch.cdist(torch.from_numpy(np.asarray(ca_l, np.float32)), torch.from_numpy(np.asarray(ca_r, np.float32)))
    idx = (d < PAIR_CUT).nonzero()
    if len(idx) == 0:
        idx = (d == d.min()).nonzero()[:1]
    dist = d[idx[:, 0], idx[:, 1]]
    if len(dist) > PAIR_MAX:
        keep = torch.topk(dist, PAIR_MAX, largest=False).indices
        idx, dist = idx[keep], dist[keep]
    return idx[:, 0].long(), idx[:, 1].long(), (dist < CONTACT).float(), d


def build_batch(lig, rec):
    """lig / rec: dicts with seq, emb, ca, muts. Returns a one-complex Batch and the full distance matrix."""
    x_l, e_l = build_side(lig["seq"], lig["emb"], lig["ca"], lig["muts"])
    x_r, e_r = build_side(rec["seq"], rec["emb"], rec["ca"], rec["muts"])
    pl, pr, pc, dist = interface_pairs(lig["ca"], rec["ca"])
    item = PairData(x_l=x_l, edge_index_l=e_l, x_r=x_r, edge_index_r=e_r, pair_l=pl, pair_r=pr, pair_y=pc)
    return Batch.from_data_list([item], follow_batch=FOLLOW), dist


@torch.no_grad()
def predict(model, meta, batch):
    """Calibrated pKd, pair attention weights and contact logits."""
    out, contact_logit, w = model(batch)
    raw = float(out) * meta["y_std"] + meta["y_mean"]
    return meta["slope"] * raw + meta["intercept"], w.numpy(), contact_logit.numpy()


def integrated_gradients(model, meta, batch, steps=100):
    """Integrated Gradients over all node features (zero baseline, midpoint rule).

    Returns one score per residue (ligand residues first, then receptor residues) in calibrated pKd units
    and the completeness error (how far the scores are from adding up to the prediction change).
    """
    xl, xr = batch.x_l, batch.x_r
    scale = meta["y_std"] * meta["slope"]

    def f(a, b_):
        return model(batch, a, b_)[0].sum() * scale

    gl, gr = torch.zeros_like(xl), torch.zeros_like(xr)
    for k in range(1, steps + 1):
        a = (k - 0.5) / steps
        xl_k, xr_k = (a * xl).requires_grad_(True), (a * xr).requires_grad_(True)
        g1, g2 = torch.autograd.grad(f(xl_k, xr_k), [xl_k, xr_k])
        gl += g1
        gr += g2
    al, ar = xl * gl / steps, xr * gr / steps
    with torch.no_grad():
        delta = float(f(xl, xr) - f(torch.zeros_like(xl), torch.zeros_like(xr)))
    total = float(al.sum() + ar.sum())
    scores = torch.cat([al.sum(1), ar.sum(1)]).detach().numpy()
    return scores, abs(delta - total) / (abs(delta) + 1e-6)
