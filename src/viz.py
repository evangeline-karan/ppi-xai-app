"""Plots, tables and the 3D viewer HTML. Needs only numpy, pandas and matplotlib."""
import json
import re

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STRONG_PKD = 7.0   # pKd >= 7 means Kd <= 100 nM
_LABEL = re.compile(r"^(.)(-?\d+)([A-Z])$")


def parse_label(label):
    m = _LABEL.match(label)
    return (m.group(1), int(m.group(2)), m.group(3)) if m else None


def _ramp_hex(t):
    """Light grey (t=0) to red (t=1)."""
    t = float(np.clip(t, 0, 1)) ** 0.5
    lo, hi = np.array([230, 230, 230]), np.array([215, 25, 28])
    r, g, b = (lo + (hi - lo) * t).astype(int)
    return f"#{r:02x}{g:02x}{b:02x}"


def viewer_html(pdb_text, res, height=520):
    """3Dmol.js viewer: cartoon coloured by |IG| (grey to red), interface residues as sticks, mutations in magenta."""
    ig = res.get("ig")
    imp = np.zeros(len(res["labels"])) if ig is None else np.abs(np.asarray(ig, float))
    imp = imp / (imp.max() + 1e-12)
    mutated = set(res["mut_nodes"])
    entries = []
    for n, label in enumerate(res["labels"]):
        p = parse_label(label)
        if p is not None:
            entries.append([p[0], p[1], _ramp_hex(imp[n]), bool(res["is_interface"][n]), n in mutated])
    return f"""
<div id="viewer" style="width:100%;height:{height}px;position:relative;"></div>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<script>
const pdb = {json.dumps(pdb_text)};
const entries = {json.dumps(entries)};
const v = $3Dmol.createViewer("viewer", {{backgroundColor: "white"}});
v.addModel(pdb, "pdb");
v.setStyle({{}}, {{cartoon: {{color: "#e6e6e6"}}}});
entries.forEach(e => {{
  const sel = {{chain: e[0], resi: e[1]}};
  v.setStyle(sel, {{cartoon: {{color: e[2]}}}});
  if (e[3]) v.addStyle(sel, {{stick: {{color: e[2], radius: 0.12}}}});
  if (e[4]) v.addStyle(sel, {{stick: {{color: "magenta", radius: 0.25}}, sphere: {{color: "magenta", radius: 0.7}}}});
}});
v.zoomTo();
v.render();
</script>"""


def residue_table(res, k=15):
    ig = res.get("ig")
    if ig is None:
        return None
    ig = np.asarray(ig, float)
    mutated = set(res["mut_nodes"])
    order = np.argsort(-np.abs(ig))[:k]
    return pd.DataFrame({
        "Residue": [res["labels"][n] for n in order],
        "IG attribution (pKd)": [round(float(ig[n]), 3) for n in order],
        "Occlusion change (pKd)": [None if np.isnan(res["occlusion"][n]) else round(float(res["occlusion"][n]), 3)
                                   for n in order],
        "At interface": [bool(res["is_interface"][n]) for n in order],
        "Mutated": [n in mutated for n in order]})


def pair_table(res, k=10):
    w = np.asarray(res["pair_attention"], float)
    order = np.argsort(-w)[:k]
    n_lig = res["n_lig"]
    return pd.DataFrame({
        "Ligand residue": [res["labels"][res["pair_l"][i]] for i in order],
        "Receptor residue": [res["labels"][n_lig + res["pair_r"][i]] for i in order],
        "Attention (relative)": [round(float(w[i] / w.max()), 3) for i in order],
        "Contact (< 8 A)": [bool(res["pair_contact"][i] > 0.5) for i in order]})


def importance_figure(res, k=15):
    ig = res.get("ig")
    if ig is None:
        return None
    ig = np.asarray(ig, float)
    mutated = set(res["mut_nodes"])
    order = np.argsort(-np.abs(ig))[:k][::-1]
    colors = ["magenta" if n in mutated else ("#e08214" if res["is_interface"][n] else "#999999") for n in order]
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.barh([res["labels"][n] for n in order], ig[order], color=colors)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("Integrated Gradients attribution (change in pKd)")
    ax.set_title("Most influential residues")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ("magenta", "#e08214", "#999999")]
    ax.legend(handles, ["mutated", "at interface", "elsewhere"], loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def pair_map_figure(res):
    w = np.asarray(res["pair_attention"], float)
    contact = np.asarray(res["pair_contact"]) > 0.5
    pl, pr = np.asarray(res["pair_l"]), np.asarray(res["pair_r"])
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    sc = ax.scatter(pr, pl, c=w / w.max(), s=8, cmap="viridis")
    if contact.any():
        ax.scatter(pr[contact], pl[contact], s=22, facecolors="none", edgecolors="red", linewidths=0.5,
                   label="contact (< 8 A)")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("Receptor residue index")
    ax.set_ylabel("Ligand residue index")
    ax.set_title("Interface pair attention")
    fig.colorbar(sc, ax=ax, label="attention (relative)")
    fig.tight_layout()
    return fig
