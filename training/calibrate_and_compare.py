import os, json, numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr

root = "/content/drive/MyDrive/ppi-xai-app"
assert os.path.exists(root), "Drive not mounted - run the mount cell first"

def load(phase, split):
    return pd.read_csv(f"{root}/results/phase{phase}_{split}_predictions.csv")

def pooled(d, col):
    y, p = d["pKd"].values, d[col].values
    return dict(rmse=float(np.sqrt(np.mean((y - p) ** 2))), mae=float(np.mean(np.abs(y - p))),
                pearson=float(pearsonr(y, p)[0]), spearman=float(spearmanr(y, p)[0]),
                r2=float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)))

def within(d, col):
    g = d.groupby("PDB")
    yc, pc = d["pKd"] - g["pKd"].transform("mean"), d[col] - g[col].transform("mean")
    m = g["pKd"].transform("size") >= 5
    per = [spearmanr(x["pKd"], x[col])[0] for _, x in d[m].groupby("PDB") if x["pKd"].std() >= 0.3]
    return dict(within_pearson=float(pearsonr(yc[m], pc[m])[0]),
                within_spearman=float(spearmanr(yc[m], pc[m])[0]),
                per_pdb_spearman=float(np.nanmean(per)))

calib, rows = {}, []
for phase in (1, 2):
    v, t = load(phase, "val"), load(phase, "test")
    slope, intercept = np.polyfit(v["pred"], v["pKd"], 1)          # fitted on VALIDATION only
    calib[f"phase{phase}"] = dict(slope=float(slope), intercept=float(intercept))
    t["pred_cal"] = slope * t["pred"] + intercept
    t.to_csv(f"{root}/results/phase{phase}_test_predictions_calibrated.csv", index=False)
    for label, col in [("raw", "pred"), ("calibrated", "pred_cal")]:
        rows.append(dict(model=f"Phase {phase}", version=label, **pooled(t, col), **within(t, col)))

with open(f"{root}/results/calibration.json", "w") as f:
    json.dump(calib, f, indent=2)

tab = pd.DataFrame(rows)
print(tab.round(3).to_string(index=False))
print("\nCalibration coefficients (fitted on validation, saved to results/calibration.json):")
print(json.dumps(calib, indent=2))
