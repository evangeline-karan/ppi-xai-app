"""Live mode: predict a user-supplied complex. Needs torch, torch_geometric, fair-esm and about 4 GB of RAM."""
import numpy as np
import requests
import torch

from .structure import node_labels, parse_mutations, parse_structure, resolve_mutations, side_offsets

MAX_RESIDUES = 1500      # keeps memory and time reasonable on a CPU server
ESM_WINDOW, ESM_STRIDE = 1000, 500


def fetch_pdb(pdb_id):
    pdb_id = pdb_id.strip().upper()
    r = requests.get(f"https://files.rcsb.org/download/{pdb_id}.pdb", timeout=60)
    if r.status_code != 200:
        raise ValueError(f"Could not download {pdb_id}.pdb from the PDB (status {r.status_code}). "
                         "Very large entries only exist in mmCIF format: download a PDB-format file and upload it instead.")
    return r.text


def load_esm():
    import esm
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.eval()
    return model, alphabet


@torch.no_grad()
def embed(esm_bundle, seq):
    """ESM-2 650M last-layer residue embeddings, shape [len(seq), 1280]. Long chains use overlapping windows."""
    model, alphabet = esm_bundle
    convert = alphabet.get_batch_converter()

    def one(s):
        _, _, toks = convert([("x", s)])
        rep = model(toks, repr_layers=[33])["representations"][33]
        return rep[0, 1:len(s) + 1].numpy()

    if len(seq) <= 1022:
        return one(seq)
    acc, cnt = np.zeros((len(seq), 1280), np.float32), np.zeros(len(seq), np.float32)
    for st in range(0, len(seq), ESM_STRIDE):
        en = min(st + ESM_WINDOW, len(seq))
        acc[st:en] += one(seq[st:en])
        cnt[st:en] += 1
        if en == len(seq):
            break
    return acc / cnt[:, None]


def run_live(pdb_text, name, lig_chains, rec_chains, mutation_text, esm_bundle, model, meta, do_ig=True, ig_steps=100):
    from .model import build_batch, integrated_gradients, predict

    chains = parse_structure(pdb_text)
    lig = list(dict.fromkeys(c.strip() for c in lig_chains if c.strip()))
    rec = list(dict.fromkeys(c.strip() for c in rec_chains if c.strip()))
    if not lig or not rec:
        raise ValueError("Enter at least one ligand chain and one receptor chain.")
    if set(lig) & set(rec):
        raise ValueError("A chain cannot be on both sides.")
    missing = [c for c in lig + rec if c not in chains]
    if missing:
        raise ValueError(f"Chain(s) {', '.join(missing)} not found. Chains with amino acids in this file: "
                         f"{', '.join(sorted(chains))}.")

    off_l, n_l = side_offsets(chains, lig)
    off_r, n_r = side_offsets(chains, rec)
    if n_l + n_r > MAX_RESIDUES:
        raise ValueError(f"This complex has {n_l + n_r} residues; live mode is limited to {MAX_RESIDUES}.")

    resolved = resolve_mutations(chains, parse_mutations(mutation_text))
    muts_l, muts_r, mut_nodes = [], [], []
    for chain, i, wt, mt in resolved:
        if chain in off_l:
            muts_l.append((off_l[chain] + i, wt, mt))
            mut_nodes.append(off_l[chain] + i)
        elif chain in off_r:
            muts_r.append((off_r[chain] + i, wt, mt))
            mut_nodes.append(n_l + off_r[chain] + i)
        else:
            raise ValueError(f"Mutation on chain {chain}, which is not one of the chains you selected.")

    cache = {}

    def side(order, muts):
        seq = "".join(chains[c]["seq"] for c in order)
        parts = []
        for c in order:
            s = chains[c]["seq"]
            if s not in cache:
                cache[s] = embed(esm_bundle, s)
            parts.append(cache[s])
        ca = np.concatenate([chains[c]["ca"] for c in order])
        return dict(seq=seq, emb=np.concatenate(parts), ca=ca, muts=muts)

    batch, dist = build_batch(side(lig, muts_l), side(rec, muts_r))
    pred, attn, _ = predict(model, meta, batch)

    is_iface = np.concatenate([(dist.min(1).values < 8.0).numpy(), (dist.min(0).values < 8.0).numpy()])
    ig, completeness = (None, None)
    if do_ig:
        ig, completeness = integrated_gradients(model, meta, batch, steps=ig_steps)
    return dict(PDB=name, mutations=mutation_text.strip() or None, true_pKd=None, pred_pKd=float(pred),
                lig_chains=lig, rec_chains=rec, n_lig=n_l,
                labels=node_labels(chains, lig) + node_labels(chains, rec),
                ig=ig, occlusion=np.full(n_l + n_r, np.nan, np.float32), is_interface=is_iface,
                mut_nodes=mut_nodes, pair_l=batch.pair_l.numpy(), pair_r=batch.pair_r.numpy(),
                pair_attention=attn, pair_contact=batch.pair_y.numpy(), completeness_error=completeness)
