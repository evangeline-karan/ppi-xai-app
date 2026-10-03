import os, json, pickle, shutil
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score

root = "/content/drive/MyDrive/ppi-xai-app"
assert os.path.exists(root), "Drive not mounted - run the mount cell first"
res_dir = f"{root}/results"
assets = f"{root}/app_assets"
os.makedirs(assets, exist_ok=True)

# ---------- 1. calibration and model weights ----------
cal = json.load(open(f"{res_dir}/calibration.json"))["phase2"]
slope, intercept = cal["slope"], cal["intercept"]
ckpt = torch.load(f"{root}/checkpoints/phase2_best.pt", map_location="cpu")
const = slope * ckpt["y_mean"] + intercept        # turns the XAI 'base' value into a real calibrated pKd
torch.save({"model": ckpt["model"], "y_mean": ckpt["y_mean"], "y_std": ckpt["y_std"],
            "slope": slope, "intercept": intercept, "epoch": ckpt["epoch"]},
           f"{assets}/phase2_weights.pt")
print(f"Weights exported (epoch {ckpt['epoch']}), calibration slope {slope:.3f}, intercept {intercept:.3f}")

# ---------- 2. binder AUROC (strong vs weak binders, pKd >= 7, i.e. Kd <= 100 nM) ----------
THR = 7.0
auc = {}
for phase in (1, 2):
    t = pd.read_csv(f"{res_dir}/phase{phase}_test_predictions_calibrated.csv")
    auc[f"phase{phase}"] = dict(auroc=float(roc_auc_score(t["pKd"] >= THR, t["pred_cal"])),
                                strong_binder_share=float((t["pKd"] >= THR).mean()))
    print(f"Phase {phase}: binder AUROC (pKd >= {THR}) = {auc[f'phase{phase}']['auroc']:.3f} "
          f"| strong-binder share in test set {auc[f'phase{phase}']['strong_binder_share']:.0%}")

# ---------- 3. demo explanations with real predicted pKd and the true value ----------
t2 = pd.read_csv(f"{res_dir}/phase2_test_predictions_calibrated.csv")
with open(f"{res_dir}/xai_examples.pkl", "rb") as f:
    examples = pickle.load(f)

demo, diffs = [], []
for ex in examples:
    cand = t2[t2["PDB"] == ex["PDB"]]
    cand = cand[cand["Mutations"].isna()] if pd.isna(ex["Mutations"]) else cand[cand["Mutations"] == ex["Mutations"]]
    target = ex["pred"] + const
    if len(cand) == 0:
        continue
    row = cand.iloc[(cand["pred_cal"] - target).abs().argmin()]
    diffs.append(abs(row["pred_cal"] - target))
    demo.append(dict(PDB=ex["PDB"], mutations=None if pd.isna(ex["Mutations"]) else ex["Mutations"],
                     true_pKd=float(row["pKd"]), pred_pKd=float(row["pred_cal"]),
                     lig_chains=ex["lig"], rec_chains=ex["rec"], n_lig=int(ex["nL"]),
                     labels=ex["labels"], ig=ex["ig"].astype(np.float32), occlusion=ex["occ"].astype(np.float32),
                     is_interface=ex["is_interface"], mut_nodes=ex["mut_nodes"],
                     pair_l=ex["pair_l"], pair_r=ex["pair_r"], pair_attention=ex["attn"].astype(np.float32),
                     pair_contact=ex["pair_y"]))
print(f"\nExported {len(demo)} of {len(examples)} demo complexes")
print(f"Sanity check, |stored prediction - recomputed calibrated prediction|: max {max(diffs):.4f} (should be ~0)")
with open(f"{assets}/demo_examples.pkl", "wb") as f:
    pickle.dump(demo, f)

# ---------- 4. one summary file for the README and the paper ----------
xai = pd.read_csv(f"{res_dir}/xai_per_complex.csv")
summary = dict(
    dataset=dict(raw_rows=12062, final_rows=11703, final_pdbs=3019, train=9355, val=1138, test=1210),
    phase1=json.load(open(f"{res_dir}/phase1_metrics.json")),
    phase2=json.load(open(f"{res_dir}/phase2_metrics.json")),
    calibration=json.load(open(f"{res_dir}/calibration.json")),
    binder_auroc=auc,
    xai=dict(n_complexes=int(len(xai)),
             structural_agreement_ig=float(xai.auc_ig.mean()), structural_agreement_occlusion=float(xai.auc_occ.mean()),
             structural_agreement_pair_attention=float(xai.auc_attn_pairs.mean()),
             faithfulness_top5_ig=float(xai.dAff_top_ig.mean()), faithfulness_top5_occlusion=float(xai.dAff_top_occ.mean()),
             faithfulness_random5=float(xai.dAff_random.mean()), stability=float(xai.stability.mean()),
             mutated_residue_percentile=float(xai.mut_ig_percentile.mean()),
             ig_completeness_error=float(xai.completeness_err.mean())))
with open(f"{assets}/final_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# ---------- 5. package ----------
zip_path = shutil.make_archive(f"{root}/ppi_xai_app_assets", "zip", assets)
print("\nFiles in app_assets:")
for n in sorted(os.listdir(assets)):
    print(f"  {n:24s} {os.path.getsize(f'{assets}/{n}') / 1e6:6.1f} MB")
print(f"\nZip for download: {zip_path} ({os.path.getsize(zip_path) / 1e6:.1f} MB)")
print("GitHub limit is 100 MB per file - all files above must be below that.")
