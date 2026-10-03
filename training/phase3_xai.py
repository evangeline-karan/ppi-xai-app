!pip -q install torch_geometric
import os, time, subprocess, pickle, json, copy
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F
from tqdm.auto import tqdm
from scipy.stats import spearmanr
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
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
assert device == "cuda", "No GPU - use Runtime > Change runtime type > T4 GPU, then run again"

# ---------- 2. load data ----------
test_df = pd.read_pickle(f"{out}/final_test.pkl")
with open(f"{out}/structures.pkl", "rb") as f:
    structures = pickle.load(f)
emb = {}
for name in tqdm(sorted(os.listdir(f"{out}/esm2"))):
    if name.endswith(".pkl"):
        with open(f"{out}/esm2/{name}", "rb") as f:
            emb.update(pickle.load(f))
print(len(test_df), "test rows |", len(emb), "embeddings loaded (expect 8432)")

# ---------- 3. feature code (identical to Phase 2) ----------
AA = "ACDEFGHIKLMNPQRSTVWY"
aa2i = {a: i for i, a in enumerate(AA)}
hydro = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4, H=-3.2, I=4.5,
             L=3.8, K=-3.9, M=1.9, F=2.8, P=-1.6, S=-0.8, T=-0.7, W=-0.9, Y=-1.3, V=4.2)
vol = dict(A=88.6, R=173.4, N=114.1, D=111.1, C=108.5, Q=143.8, E=138.4, G=60.1, H=153.2, I=166.7,
           L=166.7, K=168.6, M=162.9, F=189.9, P=112.7, S=89.0, T=116.1, W=227.8, Y=193.6, V=140.0)
charge = {a: 0.0 for a in AA}; charge.update(K=1, R=1, D=-1, E=-1, H=0.1)
PHYS = np.array([[hydro[a] / 4.5, (vol[a] - 130) / 50, charge[a],
                  float(a in "FWY"), float(a in "STNQ")] for a in AA], dtype=np.float32)
MUT_FLAG, CUTOFF = 1305, 8.0
PAIR_CUT, PAIR_MAX, CONTACT = 15.0, 3000, 8.0

edge_cache, pair_cache = {}, {}
def side_edges(pid, chains, ca):
    key = (pid, tuple(chains))
    if key not in edge_cache:
        c = torch.from_numpy(ca); d = torch.cdist(c, c)
        mask = (d < CUTOFF) & ~torch.eye(len(ca), dtype=torch.bool)
        edge_cache[key] = mask.nonzero().t().contiguous().to(torch.int32)
    return edge_cache[key].long()

def interface_pairs(pid, lig, rec):
    key = (pid, tuple(lig), tuple(rec))
    if key not in pair_cache:
        cl = torch.from_numpy(np.concatenate([structures[pid][c]["ca"] for c in lig]))
        cr = torch.from_numpy(np.concatenate([structures[pid][c]["ca"] for c in rec]))
        d = torch.cdist(cl, cr)
        idx = (d < PAIR_CUT).nonzero()
        if len(idx) == 0:
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
            blk[n, 0] = 1; blk[n, 1 + aa2i[wt]] = 1; blk[n, 21 + aa2i[mt]] = 1
            blk[n, 41:46] = PHYS[aa2i[mt]] - PHYS[aa2i[wt]]
    x = np.concatenate([esm_x, onehot, PHYS[idx], blk], axis=1)
    return torch.from_numpy(x), side_edges(pid, chains, ca)

from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.utils import softmax as pyg_softmax

class PairData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key in ("edge_index_l", "pair_l"): return self.x_l.size(0)
        if key in ("edge_index_r", "pair_r"): return self.x_r.size(0)
        return super().__inc__(key, value, *args, **kwargs)

def make_item(r):
    pid = r["PDB"].upper()
    lig = list(dict.fromkeys(r["lig_chains"])); rec = list(dict.fromkeys(r["rec_chains"]))
    x_l, e_l = build_side(pid, lig, r["mut_idx"]); x_r, e_r = build_side(pid, rec, r["mut_idx"])
    pl, pr, pc = interface_pairs(pid, lig, rec)
    return PairData(x_l=x_l, edge_index_l=e_l, x_r=x_r, edge_index_r=e_r, pair_l=pl, pair_r=pr, pair_y=pc,
                    y=torch.tensor([r["pKd"]], dtype=torch.float32))

# ---------- 4. model (same weights as Phase 2; forward can take replacement node features) ----------
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

ckpt = torch.load(f"{root}/checkpoints/phase2_best.pt", map_location=device)
model = Phase2().to(device)
model.load_state_dict(ckpt["model"]); model.eval()
y_mean, y_std = ckpt["y_mean"], ckpt["y_std"]
with open(f"{root}/results/calibration.json") as f:
    SLOPE = json.load(f)["phase2"]["slope"]
print(f"Loaded Phase 2 checkpoint from epoch {ckpt['epoch']} | calibration slope {SLOPE:.2f}")

def pkd(b, xl=None, xr=None):
    """Prediction in calibrated pKd units, up to an additive constant (differences are what we use)."""
    out, _, _ = model(b, xl, xr)
    return out * y_std * SLOPE

# ---------- 5. explanation methods ----------
def integrated_gradients(b, xl, xr, steps=200):
    """IG over all node features, zero baseline, midpoint rule. Returns attributions shaped like xl and xr."""
    gl, gr = torch.zeros_like(xl), torch.zeros_like(xr)
    for k in range(1, steps + 1):
        a = (k - 0.5) / steps
        xl_k, xr_k = (a * xl).requires_grad_(True), (a * xr).requires_grad_(True)
        g1, g2 = torch.autograd.grad(pkd(b, xl_k, xr_k).sum(), [xl_k, xr_k])
        gl += g1; gr += g2
    return xl * gl / steps, xr * gr / steps

def residue_ig(b, xl, xr, steps=200):
    al, ar = integrated_gradients(b, xl, xr, steps)
    return torch.cat([al.sum(1), ar.sum(1)]).detach().cpu().numpy(), float(al.sum() + ar.sum())

@torch.no_grad()
def occlude(b, xl, xr, nodes):
    """Zero out the features of the given residues (global node ids: ligand first, then receptor)."""
    nL = xl.size(0)
    xl2, xr2 = xl.clone(), xr.clone()
    for n in nodes:
        if n < nL: xl2[n] = 0
        else: xr2[n - nL] = 0
    return float(pkd(b, xl2, xr2))

def side_offsets(pid, chains):
    off, o = {}, 0
    for c in chains:
        off[c] = o; o += len(structures[pid][c]["seq"])
    return off, o

def node_label(pid, lig, rec, n):
    nL = side_offsets(pid, lig)[1]
    chains, k = (lig, n) if n < nL else (rec, n - nL)
    for c in chains:
        L = len(structures[pid][c]["seq"])
        if k < L:
            return f"{c}{structures[pid][c]['resnums'][k]}{structures[pid][c]['seq'][k]}"
        k -= L

# ---------- 6. choose test complexes ----------
rng = np.random.RandomState(42)
N_PER_GROUP, MAX_RES, TOPK = 30, 1500, 5
def total_size(r):
    pid = r["PDB"].upper()
    return sum(len(structures[pid][c]["seq"]) for c in dict.fromkeys(list(r["lig_chains"]) + list(r["rec_chains"])))

chosen = []
for want_mut in (True, False):
    pool = test_df[(test_df["mut_list"].apply(len) > 0) == want_mut]
    seen = set()
    for i in rng.permutation(len(pool)):
        r = pool.iloc[i]
        if r["PDB"] in seen or total_size(r) > MAX_RES:
            continue
        seen.add(r["PDB"]); chosen.append(r)
        if len(seen) == N_PER_GROUP:
            break
sel_df = pd.DataFrame(chosen).reset_index(drop=True)
print(f"Selected {len(sel_df)} test complexes ({(sel_df['mut_list'].apply(len) > 0).sum()} mutants, "
      f"{(sel_df['mut_list'].apply(len) == 0).sum()} wild-type), max {MAX_RES} residues each")

# ---------- 7. run the explanations ----------
records, examples = [], []
for k in tqdm(range(len(sel_df))):
    r = sel_df.iloc[k]
    pid = r["PDB"].upper()
    lig = list(dict.fromkeys(r["lig_chains"])); rec = list(dict.fromkeys(r["rec_chains"]))
    b = Batch.from_data_list([make_item(r)], follow_batch=["x_l", "x_r", "pair_l"]).to(device)
    xl, xr = b.x_l, b.x_r
    nL, nR = xl.size(0), xr.size(0)
    N = nL + nR

    off_l, _ = side_offsets(pid, lig); off_r, _ = side_offsets(pid, rec)
    mut_nodes = []
    for m in r["mut_idx"]:
        if m[0] in off_l: mut_nodes.append(off_l[m[0]] + m[1])
        elif m[0] in off_r: mut_nodes.append(nL + off_r[m[0]] + m[1])

    # interface residues: C-alpha within CONTACT of the other side (used ONLY to validate explanations)
    cl = torch.from_numpy(np.concatenate([structures[pid][c]["ca"] for c in lig]))
    cr = torch.from_numpy(np.concatenate([structures[pid][c]["ca"] for c in rec]))
    d = torch.cdist(cl, cr)
    is_iface = np.concatenate([(d.min(1).values < CONTACT).numpy(), (d.min(0).values < CONTACT).numpy()])

    with torch.no_grad():
        base = float(pkd(b))
        _, cl_logit, w_attn = model(b)
    w_attn, pair_y = w_attn.cpu().numpy(), b.pair_y.cpu().numpy()

    # Integrated Gradients (clean) + completeness check
    ig, ig_sum = residue_ig(b, xl, xr)
    with torch.no_grad():
        delta = base - float(pkd(b, torch.zeros_like(xl), torch.zeros_like(xr)))
    completeness_err = abs(delta - ig_sum) / (abs(delta) + 1e-6)
    ig_imp = np.abs(ig)

    # IG stability under small input noise
    sigma = 0.05 * float(torch.cat([xl, xr]).std())
    stab = []
    for _ in range(3):
        ig_n, _ = residue_ig(b, xl + sigma * torch.randn_like(xl), xr + sigma * torch.randn_like(xr), steps=64)
        stab.append(spearmanr(ig_imp, np.abs(ig_n))[0])

    # residue occlusion: interface residues + mutated residues + 50 random non-interface residues
    non_iface = np.where(~is_iface)[0]
    extra = rng.choice(non_iface, size=min(50, len(non_iface)), replace=False) if len(non_iface) else []
    cand = sorted(set(np.where(is_iface)[0].tolist()) | set(mut_nodes) | set(int(x) for x in extra))
    occ = np.full(N, np.nan)
    for n in cand:
        occ[n] = occlude(b, xl, xr, [n]) - base
    ev = np.where(~np.isnan(occ))[0]

    # faithfulness: occlude the top-k residues together, compare with k random residues
    def delta_for(nodes): return abs(occlude(b, xl, xr, list(nodes)) - base)
    top_ig = np.argsort(-ig_imp)[:TOPK]
    top_occ = ev[np.argsort(-np.abs(occ[ev]))[:TOPK]]
    rand = np.mean([delta_for(rng.choice(N, TOPK, replace=False)) for _ in range(10)])

    def auc(score, label):
        return float(roc_auc_score(label, score)) if 0 < label.sum() < len(label) else np.nan

    rec_ = dict(PDB=pid, mutated=len(mut_nodes) > 0, n_res=N, n_interface=int(is_iface.sum()),
                pred=base, completeness_err=completeness_err,
                auc_ig=auc(ig_imp, is_iface), auc_occ=auc(np.abs(occ[ev]), is_iface[ev]),
                auc_attn_pairs=auc(w_attn, pair_y),
                dAff_top_ig=delta_for(top_ig), dAff_top_occ=delta_for(top_occ), dAff_random=float(rand),
                stability=float(np.nanmean(stab)))
    if mut_nodes:
        ranks = [(ig_imp > ig_imp[n]).sum() / (N - 1) for n in mut_nodes]       # 0 = most important
        rec_["mut_ig_percentile"] = float(1 - np.mean(ranks))                    # 1 = most important
    records.append(rec_)
    examples.append(dict(PDB=pid, Mutations=r["Mutations"], lig=lig, rec=rec, nL=nL, ig=ig, occ=occ,
                         is_interface=is_iface, mut_nodes=mut_nodes, attn=w_attn,
                         pair_l=b.pair_l.cpu().numpy(), pair_r=b.pair_r.cpu().numpy(), pair_y=pair_y,
                         labels=[node_label(pid, lig, rec, n) for n in range(N)], pred=base))

res = pd.DataFrame(records)
res.to_csv(f"{root}/results/xai_per_complex.csv", index=False)
with open(f"{root}/results/xai_examples.pkl", "wb") as f:
    pickle.dump(examples, f)

# ---------- 8. summary ----------
def ms(x): return f"{np.nanmean(x):.3f} (median {np.nanmedian(x):.3f}, n={int(np.sum(~np.isnan(x)))})"
print("\n=== XAI VALIDATION on", len(res), "held-out test complexes (Phase 2 model, calibrated pKd units) ===")
print("Structural agreement = AUROC of importance for finding interface residues (0.5 = chance):")
print("   Integrated Gradients :", ms(res.auc_ig))
print("   Residue occlusion    :", ms(res.auc_occ))
print("   Pair attention (contact pairs):", ms(res.auc_attn_pairs))
print(f"Faithfulness (mean |change in predicted pKd| after masking {TOPK} residues):")
print(f"   top-{TOPK} by IG: {res.dAff_top_ig.mean():.3f} | top-{TOPK} by occlusion: {res.dAff_top_occ.mean():.3f} "
      f"| {TOPK} random residues: {res.dAff_random.mean():.3f}")
print(f"   share of complexes where top-{TOPK} (IG) beats random: {(res.dAff_top_ig > res.dAff_random).mean():.0%}")
print("Stability (Spearman of IG importance, clean vs slightly noisy input):", ms(res.stability))
mm = res[res.mutated]
print("Mutated residue importance percentile by IG (1 = most important, 0.5 = chance):", ms(mm.mut_ig_percentile))
print("IG completeness error (should be small):", ms(res.completeness_err))
print("   share of complexes with completeness error below 10%:", f"{(res.completeness_err < 0.1).mean():.0%}")

for want in (True, False):
    ex = next(e for e in examples if bool(e["mut_nodes"]) == want)
    top = np.argsort(-np.abs(ex["ig"]))[:5]
    print(f"\nExample {ex['PDB']} ({ex['Mutations'] if want else 'wild-type'}), top-5 residues by |IG|:")
    for n in top:
        print(f"   {ex['labels'][n]:8s} IG {ex['ig'][n]:+.3f} | interface: {bool(ex['is_interface'][n])}"
              f"{' | MUTATED' if n in ex['mut_nodes'] else ''}")
print("\nSaved: results/xai_per_complex.csv and results/xai_examples.pkl")
