"""Structure and mutation helpers. These need only numpy and Biopython (no torch)."""
import re
from io import StringIO

import numpy as np
from Bio.Data.IUPACData import protein_letters_3to1
from Bio.PDB import PDBParser

AA = "ACDEFGHIKLMNPQRSTVWY"
_MUTATION = re.compile(r"^([A-Za-z0-9])_([A-Z])(-?\d+)([A-Z])$")


def parse_structure(pdb_text):
    """Read the first model of a PDB file.

    Returns {chain_id: {"resnums", "icodes", "seq", "ca"}} using the residues that have a C-alpha atom and
    are one of the 20 standard amino acids (the same rule that was used to build the training data).
    """
    structure = PDBParser(QUIET=True).get_structure("query", StringIO(pdb_text))
    model = next(structure.get_models())
    chains = {}
    for ch in model:
        resnums, icodes, seq, ca = [], [], [], []
        for res in ch:
            if res.id[0] != " " or "CA" not in res:
                continue
            aa = protein_letters_3to1.get(res.get_resname().capitalize())
            if aa is None or aa not in AA:
                continue
            resnums.append(res.id[1])
            icodes.append(res.id[2])
            seq.append(aa)
            ca.append(res["CA"].coord)
        if seq:
            chains[ch.id] = {"resnums": resnums, "icodes": icodes, "seq": "".join(seq),
                             "ca": np.array(ca, dtype=np.float32)}
    return chains


def parse_mutations(text):
    """'I_L38G, I_R46A' -> [('I','L','38','G'), ('I','R','46','A')]. Empty text means wild type."""
    muts = []
    for part in (text or "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        m = _MUTATION.match(part)
        if m is None:
            raise ValueError(f"Could not read mutation '{part}'. Use Chain_WildTypePositionMutant, e.g. I_L38G.")
        muts.append(m.groups())
    return muts


def resolve_mutations(chains, muts):
    """Check each mutation against the structure (PDB residue numbering).

    Returns [(chain, index_in_chain_sequence, wild_type, mutant)] and raises ValueError if the chain or
    position does not exist or the wild-type letter does not match.
    """
    out = []
    for chain, wt, pos, mt in muts:
        if chain not in chains:
            raise ValueError(f"Mutation {chain}_{wt}{pos}{mt}: chain {chain} is not in the structure.")
        c = chains[chain]
        index = {(n, ic): i for i, (n, ic) in enumerate(zip(c["resnums"], c["icodes"]))}
        key = (int(pos), " ")
        if key not in index:
            raise ValueError(f"Mutation {chain}_{wt}{pos}{mt}: residue {pos} was not found in chain {chain} "
                             f"(PDB numbering, residues without a C-alpha are skipped).")
        i = index[key]
        if c["seq"][i] != wt:
            raise ValueError(f"Mutation {chain}_{wt}{pos}{mt}: the structure has {c['seq'][i]} at position {pos} "
                             f"of chain {chain}, not {wt}.")
        out.append((chain, i, wt, mt))
    return out


def side_offsets(chains, order):
    """Start index of each chain when the chains in `order` are concatenated, and the total length."""
    offsets, total = {}, 0
    for c in order:
        offsets[c] = total
        total += len(chains[c]["seq"])
    return offsets, total


def node_labels(chains, order):
    """Residue labels like 'B335I' (chain, residue number, amino acid) in node order."""
    labels = []
    for c in order:
        d = chains[c]
        labels.extend(f"{c}{n}{a}" for n, a in zip(d["resnums"], d["seq"]))
    return labels
