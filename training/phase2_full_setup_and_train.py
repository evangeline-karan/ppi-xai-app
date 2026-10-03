!pip -q install torch_geometric
import os, time, subprocess, pickle, json, copy
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F
from tqdm.auto import tqdm
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

# ---------- 1. Drive + GPU (run the separate mount cell first if the mount fails here) ----------
def drive_mounted():
    r = subprocess.run("mount | grep -c '/content/drive'", shell=True, capture_output=True, text=True)
    return r.stdout.strip() != "0"

if not drive_mounted():
    if os.path.exists("/content/drive"):
        os.rename("/content/drive", f"/content/drive_stray_{int(time.time())}")
    from google.colab import drive
    drive.mount("/content/drive")
assert drive_mounted(), "Drive is not mounted - run the separate mount cell first"

root = "/content/drive/MyDrive/ppi-xai-app"
out = f"{root}/data/processed"
os.makedirs(f"{root}/checkpoints", exist_ok=True)
os.makedirs(f"{root}/results", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
assert device == "cuda", "No GPU - use Runtime > Change runtime type > T4 GPU, then run again"

# ---------- 2. load data ----------
final = pd.read_pickle(f"{out}/final_all.pkl")
train_df = pd.read_pickle(f"{out}/final_train.pkl")
val_df = pd.read_pickle(f"{out}/final_val.pkl")
test_df = pd.read_pickle(f"{out}/final_test.pkl")
with open(f"{out}/structures.pkl", "rb") as f:
    structures = pickle.load(f)
print(len(final), len(train_df), len(val_df), len(test_df), "(expect 11703 9355 1138 1210)")

emb = {}
for name in tqdm(sorted(os.listdir(f"{out}/esm2"))):
    if name.endswith(".pkl"):
        with open(f"{out}/esm2/{name}", "rb") as f:
            emb.update(pickle.load(f))
print(len(emb), "embeddings loaded (expect 8432)")

# ---------- 3. amino-acid tables (identical to Phase 1) ----------
AA = "ACDEFGHIKLMNPQRSTVWY"
aa2i = {a: i for i, a in enumerate(AA)}
hydro = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4, H=-3.2, I=4.5,
             L=3.8, K=-3.9, M=1.9, F=2.8, P=-1.6, S=-0.8, T=-0.7, W=-0.9, Y=-1.3, V=4.2)
vol = dict(A=88.6, R=173.4, N=114.1, D=111.1, C=108.5, Q=143.8, E=138.4, G=60.1, H=153.2, I=166.7,
           L=166.7, K=168.6, M=162.9, F=189.9, P=112.7, S=89.0, T=116.1, W=227.8, Y=193.6, V=140.0)
charge = {a: 0.0 for a in AA}; charge.update(K=1, R=1, D=-1, E=-1, H=0.1)
PHYS = np.array([[hydro[a] / 4.5, (vol[a] - 130) / 50, charge[a],
                  float(a in "FWY"), float(a in "STNQ")] for a in AA], dtype=np.float32)
FEAT_DIM = 1280 + 20 + 5 + 46
MUT_FLAG = 1305        # column of the "this residue is mutated" flag in the node features
CUTOFF = 8.0           # intra-chain edge cutoff (A)

# interface pair matrix settings (Phase 2)
PAIR_CUT = 15.0        # keep ligand-receptor residue pairs closer than this (A)
PAIR_MAX = 3000        # at most this many pairs per complex (nearest first)
CONTACT = 8.0          # a pair with C-alpha distance < 8 A counts as a contact (auxiliary target)

# ---------- 4. graph building ----------
edge_cache, pair_cache = {}, {}

def side_edges(pid, chains, ca):
    key = (pid, tuple(chains))
    if key not in edge_cache:
        c = torch.from_numpy(ca)
        d = torch.cdist(c, c)
        mask = (d < CUTOFF) & ~torch.eye(len(ca), dtype=torch.bool)
        edge_cache[key] = mask.nonzero().t().contiguous().to(torch.int32)
    return edge_cache[key].long()

def interface_pairs(pid, lig, rec):
    """Sparse residue-pair matrix: all ligand-receptor pairs closer than PAIR_CUT (nearest PAIR_MAX kept)."""
    key = (pid, tuple(lig), tuple(rec))
    if key not in pair_cache:
        cl = torch.from_numpy(np.concatenate([structures[pid][c]["ca"] for c in lig]))
        cr = torch.from_numpy(np.concatenate([structures[pid][c]["ca"] for c in rec]))
        d = torch.cdist(cl, cr)
        idx = (d < PAIR_CUT).nonzero()
        if len(idx) == 0:                                   # safety: take the single nearest pair
            idx = (d == d.min()).nonzero()[:1]
        dist = d[idx[:, 0], idx[:, 1]]
        if len(dist) > PAIR_MAX:
            keep = torch.topk(dist, PAIR_MAX, largest=False).indices
            idx, dist = idx[keep], dist[keep]
        pair_cache[key] = (idx[:, 0].int(), idx[:, 1].int(), (dist < CONTACT).float())
    pl, pr, contact = pair_cache[key]
    return pl.long(), pr.long(), contact

def build_side(pid, chains, muts):
    seqs = [structures[pid][c]["seq"] for c in chains]
    esm_x = np.concatenate([emb[s] for s in seqs]).astype(np.float32)
    idx = np.array([aa2i[a] for a in "".join(seqs)])
    onehot = np.eye(20, dtype=np.float32)[idx]
    ca = np.concatenate([structures[pid][c]["ca"] for c in chains])
    offsets, o = {}, 0
    for c, s in zip(chains, seqs):
        offsets[c] = o; o += len(s)
    blk = np.zeros((len(idx), 46), np.float32)
    for m in muts:
        chain, i, wt, mt = m[:4]
        if chain in offsets:
            n = offsets[chain] + i
            blk[n, 0] = 1
            blk[n, 1 + aa2i[wt]] = 1
            blk[n, 21 + aa2i[mt]] = 1
            blk[n, 41:46] = PHYS[aa2i[mt]] - PHYS[aa2i[wt]]
    x = np.concatenate([esm_x, onehot, PHYS[idx], blk], axis=1)
    return torch.from_numpy(x), side_edges(pid, chains, ca)

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.utils import softmax as pyg_softmax

class PairData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key in ("edge_index_l", "pair_l"): return self.x_l.size(0)
        if key in ("edge_index_r", "pair_r"): return self.x_r.size(0)
        return super().__inc__(key, value, *args, **kwargs)

class PPIDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df.reset_index()
    def __len__(self):
        return len(self.df)
    def __getitem__(self, k):
        r = self.df.iloc[k]
        pid = r["PDB"].upper()
        lig = list(dict.fromkeys(r["lig_chains"]))
        rec = list(dict.fromkeys(r["rec_chains"]))
        x_l, e_l = build_side(pid, lig, r["mut_idx"])
        x_r, e_r = build_side(pid, rec, r["mut_idx"])
        pl, pr, pc = interface_pairs(pid, lig, rec)
        return PairData(x_l=x_l, edge_index_l=e_l, x_r=x_r, edge_index_r=e_r,
                        pair_l=pl, pair_r=pr, pair_y=pc,
                        y=torch.tensor([r["pKd"]], dtype=torch.float32),
                        n_mut=torch.tensor([float(len(r["mut_idx"]))]),
                        rid=torch.tensor([int(r["index"])]))

train_ds, val_ds, test_ds = PPIDataset(train_df), PPIDataset(val_df), PPIDataset(test_df)
s = train_ds[0]
print("Quick check - ligand x:", tuple(s.x_l.shape), "| receptor x:", tuple(s.x_r.shape),
      "| interface pairs:", s.pair_l.numel(), "| contact share: %.2f" % s.pair_y.mean().item())

# ---------- 5. settings (same as Phase 1 for a fair comparison) ----------
EPOCHS = 30
BATCH = 16             # lower to 8 if you get an out-of-memory error
LR, WD, PATIENCE = 5e-4, 1e-2, 8
LAMBDA_AUX = 0.2       # weight of the contact auxiliary loss
torch.manual_seed(42); np.random.seed(42)

y_mean, y_std = float(train_df["pKd"].mean()), float(train_df["pKd"].std())
print(f"Train pKd mean {y_mean:.2f}, std {y_std:.2f}")

# ---------- 6. model ----------
class NodeEncoder(nn.Module):
    """Same GATv2 stack as Phase 1, but returns one embedding per residue (no pooling)."""
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
        self.enc_l = NodeEncoder(in_dim, hid, drop=drop)
        self.enc_r = NodeEncoder(in_dim, hid, drop=drop)
        self.proj_l, self.proj_r = nn.Linear(hid, pd), nn.Linear(hid, pd)
        # pair embedding from [a*c, |a-c|, mutated-flag of each residue]
        self.pair_mlp = nn.Sequential(nn.Linear(2 * pd + 2, pd), nn.ReLU(), nn.Dropout(drop),
                                      nn.Linear(pd, pd), nn.ReLU())
        self.contact_head = nn.Linear(pd, 1)      # auxiliary task: is this pair a contact (< 8 A)?
        self.attn_head = nn.Linear(pd, 1)         # attention score used for pooling the pairs
        self.head = nn.Sequential(nn.Linear(2 * hid + pd, 256), nn.ReLU(), nn.Dropout(drop),
                                  nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, b):
        hl, hr = self.enc_l(b.x_l, b.edge_index_l), self.enc_r(b.x_r, b.edge_index_r)
        a, c = self.proj_l(hl)[b.pair_l], self.proj_r(hr)[b.pair_r]
        mut = torch.stack([b.x_l[b.pair_l, MUT_FLAG], b.x_r[b.pair_r, MUT_FLAG]], dim=1)
        z = self.pair_mlp(torch.cat([a * c, (a - c).abs(), mut], dim=1))          # [pairs, pd]
        contact_logit = self.contact_head(z).squeeze(-1)
        pb = b.pair_l_batch                                                        # complex id of each pair
        w = pyg_softmax(self.attn_head(z).squeeze(-1), pb)                         # attention over pairs
        g = torch.zeros(b.num_graphs, z.size(1), device=z.device).index_add_(0, pb, w.unsqueeze(-1) * z)
        pooled = torch.cat([global_mean_pool(hl, b.x_l_batch), global_mean_pool(hr, b.x_r_batch), g], dim=1)
        return self.head(pooled).squeeze(-1), contact_logit, w

model2 = Phase2().to(device)
print("Parameters: %.2f M" % (sum(p.numel() for p in model2.parameters()) / 1e6))

# ---------- 7. loaders and helpers ----------
fb = ["x_l", "x_r", "pair_l"]
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, follow_batch=fb)
val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, follow_batch=fb)
test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False, follow_batch=fb)

def metrics(y, p):
    return dict(rmse=float(np.sqrt(np.mean((y - p) ** 2))), mae=float(np.mean(np.abs(y - p))),
                pearson=float(pearsonr(y, p)[0]), spearman=float(spearmanr(y, p)[0]),
                r2=float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)))

def within_metrics(meta):
    """Ranking inside one complex: remove each PDB's mean, use PDBs with >= 5 rows."""
    g = meta.groupby("PDB")
    yc, pc = meta["pKd"] - g["pKd"].transform("mean"), meta["pred"] - g["pred"].transform("mean")
    m = g["pKd"].transform("size") >= 5
    per = [spearmanr(d["pKd"], d["pred"])[0] for _, d in meta[m].groupby("PDB") if d["pKd"].std() >= 0.3]
    return dict(within_pearson=float(pearsonr(yc[m], pc[m])[0]), within_spearman=float(spearmanr(yc[m], pc[m])[0]),
                per_pdb_spearman_mean=float(np.nanmean(per)), n_pdbs=len(per))

@torch.no_grad()
def predict(model, loader):
    model.eval()
    P, Y, R, CL, CY = [], [], [], [], []
    for b in loader:
        b = b.to(device)
        out, cl, _ = model(b)
        P.append(out.cpu().numpy() * y_std + y_mean); Y.append(b.y.cpu().numpy()); R.append(b.rid.cpu().numpy())
        CL.append(cl.cpu().numpy()); CY.append(b.pair_y.cpu().numpy())
    return (np.concatenate(Y), np.concatenate(P), np.concatenate(R), np.concatenate(CL), np.concatenate(CY))

for name, d in [("Val", val_df), ("Test", test_df)]:
    print(f"{name} RMSE if you always predict the train mean: {np.sqrt(np.mean((d['pKd'] - y_mean) ** 2)):.3f}")

# ---------- 8. training ----------
opt = torch.optim.AdamW(model2.parameters(), lr=LR, weight_decay=WD)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
best, best_state, bad, history = 1e9, None, 0, []

for epoch in range(1, EPOCHS + 1):
    model2.train(); t0 = time.time(); tot = tot_aux = n = 0
    for b in tqdm(train_loader, leave=False, desc=f"epoch {epoch}"):
        b = b.to(device)
        opt.zero_grad()
        out, cl, _ = model2(b)
        l_main = F.mse_loss(out, (b.y - y_mean) / y_std)
        l_aux = F.binary_cross_entropy_with_logits(cl, b.pair_y)
        (l_main + LAMBDA_AUX * l_aux).backward()
        nn.utils.clip_grad_norm_(model2.parameters(), 1.0)
        opt.step()
        tot += l_main.item() * b.num_graphs; tot_aux += l_aux.item() * b.num_graphs; n += b.num_graphs
    yv, pv, _, clv, cyv = predict(model2, val_loader)
    mv = metrics(yv, pv)
    auc = float(roc_auc_score(cyv, clv))
    sched.step(mv["rmse"])
    train_rmse = float(np.sqrt(tot / n) * y_std)
    history.append(dict(epoch=epoch, train_rmse=train_rmse, train_aux=tot_aux / n, val_contact_auc=auc,
                        **{f"val_{k}": v for k, v in mv.items()}))
    print(f"epoch {epoch:2d} | train RMSE {train_rmse:.3f} | val RMSE {mv['rmse']:.3f} | "
          f"val Pearson {mv['pearson']:.3f} | val contact AUC {auc:.3f} | lr {opt.param_groups[0]['lr']:.1e} | {time.time() - t0:.0f}s")
    if mv["rmse"] < best:
        best, bad = mv["rmse"], 0
        best_state = copy.deepcopy(model2.state_dict())
        torch.save({"model": best_state, "y_mean": y_mean, "y_std": y_std, "epoch": epoch,
                    "val_rmse": best}, f"{root}/checkpoints/phase2_best.pt")
    else:
        bad += 1
        if bad >= PATIENCE:
            print("Early stopping"); break

pd.DataFrame(history).to_csv(f"{root}/results/phase2_history.csv", index=False)

# ---------- 9. final evaluation with the best checkpoint ----------
model2.load_state_dict(best_state)
res = {}
for name, loader in [("val", val_loader), ("test", test_loader)]:
    y, p, rid, cl, cy = predict(model2, loader)
    res[name] = metrics(y, p)
    res[name]["contact_auc"] = float(roc_auc_score(cy, cl))
    meta = final.loc[rid, ["PDB", "Mutations", "pKd"]].copy()
    meta["pred"], meta["n_mut"] = p, final.loc[rid, "mut_list"].apply(len).values
    meta.to_csv(f"{root}/results/phase2_{name}_predictions.csv", index=False)
    if name == "test":
        wt, mu = meta["n_mut"].values == 0, meta["n_mut"].values > 0
        res["test_wildtype"] = metrics(y[wt], p[wt])
        res["test_mutant"] = metrics(y[mu], p[mu])
        res["test_within_complex"] = within_metrics(meta)

print("\n=== PHASE 2 RESULTS (best-validation checkpoint) ===")
for k in ["val", "test", "test_wildtype", "test_mutant"]:
    v = res[k]
    print(f"{k:14s} RMSE {v['rmse']:.3f} | MAE {v['mae']:.3f} | Pearson {v['pearson']:.3f} | "
          f"Spearman {v['spearman']:.3f} | R2 {v['r2']:.3f}")
print("Contact AUC (auxiliary task): val %.3f | test %.3f" % (res["val"]["contact_auc"], res["test"]["contact_auc"]))
print("Within-complex (test):", {k: round(v, 3) for k, v in res["test_within_complex"].items()})

# ---------- 10. side-by-side with Phase 1 ----------
try:
    p1_meta = pd.read_csv(f"{root}/results/phase1_test_predictions.csv")
    with open(f"{root}/results/phase1_metrics.json") as f:
        p1 = json.load(f)
    p1w = within_metrics(p1_meta)
    print("\n=== PHASE 1 vs PHASE 2 (test set) ===")
    print(f"{'':10s} {'RMSE':>6s} {'Pearson':>8s} {'Spearman':>9s} {'within-Pearson':>15s} {'within-Spearman':>16s} {'per-PDB Spearman':>17s}")
    print(f"{'Phase 1':10s} {p1['test']['rmse']:6.3f} {p1['test']['pearson']:8.3f} {p1['test']['spearman']:9.3f} "
          f"{p1w['within_pearson']:15.3f} {p1w['within_spearman']:16.3f} {p1w['per_pdb_spearman_mean']:17.3f}")
    w2 = res["test_within_complex"]
    print(f"{'Phase 2':10s} {res['test']['rmse']:6.3f} {res['test']['pearson']:8.3f} {res['test']['spearman']:9.3f} "
          f"{w2['within_pearson']:15.3f} {w2['within_spearman']:16.3f} {w2['per_pdb_spearman_mean']:17.3f}")
except Exception as e:
    print("Could not build the Phase 1 comparison:", e)

with open(f"{root}/results/phase2_metrics.json", "w") as f:
    json.dump(res, f, indent=2)
