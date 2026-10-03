"""Trustworthy protein-protein binding affinity prediction with explanations (Streamlit app).

Demo mode (always on): browse 60 held-out test complexes with precomputed predictions and explanations.
Live mode (set the environment variable PPI_LIVE=1 on a server with >= 4 GB RAM): predict your own complex.
"""
import json
import os
import pickle
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

from src import viz

ASSETS = Path(__file__).parent / "app_assets"
LIVE = os.environ.get("PPI_LIVE", "0") == "1"

# Test-set results after the linear calibration fitted on the validation set (from the training notebooks).
CALIBRATED = {
    "Phase 1 (baseline)": dict(rmse=1.445, mae=1.160, pearson=0.680, spearman=0.680, r2=0.390,
                               within_pearson=0.230, within_spearman=0.098),
    "Phase 2 (used here)": dict(rmse=1.428, mae=1.122, pearson=0.680, spearman=0.672, r2=0.405,
                                within_pearson=0.206, within_spearman=0.165),
}

st.set_page_config(page_title="PPI binding affinity with explanations", layout="wide")


@st.cache_data
def load_assets():
    with open(ASSETS / "demo_examples.pkl", "rb") as f:
        demo = pickle.load(f)
    summary = {}
    if (ASSETS / "final_summary.json").exists():
        summary = json.loads((ASSETS / "final_summary.json").read_text())
    return demo, summary


@st.cache_data(show_spinner=False)
def fetch_pdb_text(pdb_id):
    try:
        r = requests.get(f"https://files.rcsb.org/download/{pdb_id}.pdb", timeout=20)
        return r.text if r.status_code == 200 else None
    except requests.RequestException:
        return None


@st.cache_resource(show_spinner="Loading ESM-2 (first time only, about 2.5 GB)...")
def get_esm():
    from src.live import load_esm
    return load_esm()


@st.cache_resource(show_spinner="Loading the model...")
def get_model():
    from src.model import load_model
    return load_model(str(ASSETS / "phase2_weights.pt"))


def kd_text(pkd):
    nm = 10 ** (9 - pkd)
    return f"{nm / 1000:.2f} uM" if nm >= 1000 else (f"{nm:.1f} nM" if nm >= 1 else f"{nm * 1000:.1f} pM")


def show_result(res, pdb_text, key):
    pred = res["pred_pKd"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted pKd", f"{pred:.2f}", help="pKd = -log10(Kd). Higher means tighter binding.")
    c2.metric("Predicted Kd", kd_text(pred))
    if res.get("true_pKd") is not None:
        c3.metric("Measured pKd", f"{res['true_pKd']:.2f}", delta=f"{pred - res['true_pKd']:+.2f} prediction error",
                  delta_color="off")
    c4.metric("Binder class", "Strong" if pred >= viz.STRONG_PKD else "Weaker",
              help="Strong means pKd >= 7, i.e. Kd of 100 nM or better.")
    st.caption("Typical error on held-out complexes is about 1.4 pKd units (RMSE 1.43), so treat single predictions "
               "as rough estimates. No uncertainty estimate is available yet.")

    left, right = st.columns(2)
    with left:
        st.subheader("Structure")
        if pdb_text:
            components.html(viz.viewer_html(pdb_text, res), height=540)
            st.caption("Red = residues the model's prediction depends on most (Integrated Gradients). "
                       "Sticks = interface residues. Magenta = mutated residues.")
        else:
            st.info("The 3D structure could not be loaded (no internet access to the PDB from this server).")
    with right:
        fig = viz.importance_figure(res)
        if fig is not None:
            st.subheader("Influential residues")
            st.pyplot(fig)
            viz.plt.close(fig)
        else:
            st.info("Residue importance was not computed for this run.")

    t1, t2 = st.columns(2)
    with t1:
        st.subheader("Interface pair attention")
        fig = viz.pair_map_figure(res)
        st.pyplot(fig)
        viz.plt.close(fig)
    with t2:
        st.subheader("Top residue pairs")
        st.dataframe(viz.pair_table(res), hide_index=True, width="stretch")
        table = viz.residue_table(res)
        if table is not None:
            st.subheader("Top residues")
            st.dataframe(table, hide_index=True, width="stretch")
    if res.get("completeness_error") is not None:
        st.caption(f"Explanation quality check: Integrated Gradients completeness error "
                   f"{res['completeness_error']:.1%} (below 10% is good).")


st.title("Protein-protein binding affinity with explanations")
st.write("A graph neural network predicts how tightly two proteins bind (pKd) and highlights which residues "
         "and residue pairs the prediction relies on. Research prototype, not for clinical use.")

try:
    demo, summary = load_assets()
except FileNotFoundError:
    st.error("The files in the app_assets folder are missing. See the README.")
    st.stop()

tab_demo, tab_live, tab_about = st.tabs(["Explore examples", "Predict your own complex", "About the model"])

with tab_demo:
    st.write("These 60 complexes were **held out from training**. Predictions and explanations were computed in advance.")
    kind = st.radio("Show", ["All", "Mutants", "Wild type"], horizontal=True)
    items = [d for d in demo if kind == "All" or (kind == "Mutants") == bool(d["mutations"])]
    names = [f"{d['PDB']} - {(d['mutations'] or 'wild type')[:40]} (measured pKd {d['true_pKd']:.2f})" for d in items]
    choice = st.selectbox("Choose a complex", range(len(items)), format_func=lambda i: names[i])
    res = items[choice]
    show_result(res, fetch_pdb_text(res["PDB"]), key="demo")

with tab_live:
    if not LIVE:
        st.info("Live prediction needs ESM-2 (about 2.6 GB of memory), which this server does not provide. "
                "It is available when the app runs on a server with enough RAM, for example a Hugging Face Space "
                "with the variable PPI_LIVE=1. See the README.")
    else:
        st.write("Enter a PDB entry (or upload a PDB file), choose which chains form each side, and optionally "
                 "list mutations as `Chain_WildTypePositionMutant` using the **PDB residue numbers**, e.g. `I_L38G, I_R46A`.")
        source = st.radio("Structure", ["PDB ID", "Upload a PDB file"], horizontal=True)
        pdb_id = st.text_input("PDB ID", "1ACB") if source == "PDB ID" else None
        upload = st.file_uploader("PDB file", type=["pdb", "ent"]) if source != "PDB ID" else None
        c1, c2 = st.columns(2)
        lig_text = c1.text_input("Ligand chain(s)", "E", help="Separate several chains with commas, e.g. A,B")
        rec_text = c2.text_input("Receptor chain(s)", "I")
        mut_text = st.text_input("Mutations (leave empty for wild type)", "I_L38G")
        do_ig = st.checkbox("Compute residue importance (Integrated Gradients, slower)", value=True)
        if st.button("Run prediction", type="primary"):
            try:
                from src.live import run_live
                with st.spinner("Reading structure, computing ESM-2 embeddings and predicting..."):
                    if source == "PDB ID":
                        from src.live import fetch_pdb
                        text, name = fetch_pdb(pdb_id), pdb_id.strip().upper()
                    else:
                        if upload is None:
                            raise ValueError("Please upload a PDB file.")
                        text, name = upload.getvalue().decode("utf-8", errors="ignore"), Path(upload.name).stem
                    model, meta = get_model()
                    out = run_live(text, name, lig_text.split(","), rec_text.split(","), mut_text,
                                   get_esm(), model, meta, do_ig=do_ig)
                st.session_state["live"] = (out, text)
            except ValueError as e:
                st.error(str(e))
            except Exception as e:  # noqa: BLE001
                st.error(f"Something went wrong: {e}")
        if "live" in st.session_state:
            show_result(*st.session_state["live"], key="live")

with tab_about:
    st.header("How good is it?")
    st.write("Numbers are on the held-out test set (1,210 samples, 302 complexes that were never used for training "
             "or model selection). Predictions are linearly calibrated using the validation set.")
    rows = []
    for name, m in CALIBRATED.items():
        auc = summary.get("binder_auroc", {}).get("phase1" if name.startswith("Phase 1") else "phase2", {})
        rows.append({"Model": name, "RMSE": m["rmse"], "MAE": m["mae"], "Pearson": m["pearson"],
                     "Spearman": m["spearman"], "R2": m["r2"],
                     "Within-complex Spearman": m["within_spearman"],
                     "Strong vs weak binder AUROC": round(auc["auroc"], 3) if auc else None})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("A model that always predicts the average would have RMSE 1.85. Within-complex Spearman measures how "
               "well mutants of the same complex are ranked.")

    xai = summary.get("xai")
    if xai:
        st.header("How trustworthy are the explanations?")
        st.write(f"Measured on {xai['n_complexes']} held-out complexes (Phase 2 model):")
        st.dataframe(pd.DataFrame([
            {"Check": "Structural agreement: AUROC for finding interface residues (0.5 = chance)",
             "Integrated Gradients": round(xai["structural_agreement_ig"], 2),
             "Occlusion": round(xai["structural_agreement_occlusion"], 2),
             "Pair attention": round(xai["structural_agreement_pair_attention"], 2)},
            {"Check": "Faithfulness: mean change in predicted pKd when the 5 top residues are masked (random: "
                      f"{xai['faithfulness_random5']:.3f})",
             "Integrated Gradients": round(xai["faithfulness_top5_ig"], 2),
             "Occlusion": round(xai["faithfulness_top5_occlusion"], 2), "Pair attention": None},
            {"Check": "Stability: rank correlation of importance under small input noise",
             "Integrated Gradients": round(xai["stability"], 2), "Occlusion": None, "Pair attention": None},
        ]), hide_index=True, width="stretch")

    st.header("Limitations")
    st.markdown(
        "- The model ranks different complexes reasonably well but is **weak at ranking mutants of the same "
        "complex** (within-complex Spearman about 0.17).\n"
        "- Explanations are faithful to the model (masking the highlighted residues changes the prediction) but only "
        "**modestly agree with the structural interface** (AUROC 0.60-0.66). The model also relies on residues far "
        "from the interface.\n"
        "- Results come from a single training run and a single dataset (PPB-Affinity); there is no uncertainty "
        "estimate yet.\n"
        "- Live mode uses PDB residue numbering and a 1,500 residue limit.\n"
        "- Research prototype. Do not use for clinical or safety-critical decisions.")
