import os, numpy as np, pandas as pd
from scipy.stats import pearsonr

root = "/content/drive/MyDrive/ppi-xai-app"
assert os.path.exists(root), "Drive not mounted - run the mount cell first"

def load(phase, split):
    return pd.read_csv(f"{root}/results/phase{phase}_{split}_predictions.csv")

print("=== Bias and scale of the predictions ===")
print(f"{'':8s}{'split':6s}{'mean y':>8s}{'mean pred':>10s}{'bias':>7s}{'std y':>7s}{'std pred':>9s}{'Pearson':>8s}{'RMSE':>7s}")
for phase in (1, 2):
    for split in ("val", "test"):
        d = load(phase, split)
        err = d.pred - d.pKd
        print(f"Phase {phase} {split:5s}{d.pKd.mean():8.2f}{d.pred.mean():10.2f}{err.mean():7.2f}"
              f"{d.pKd.std():7.2f}{d.pred.std():9.2f}{pearsonr(d.pKd, d.pred)[0]:8.3f}{np.sqrt((err**2).mean()):7.3f}")

print("\n=== Linear recalibration fitted on VALIDATION only, applied to TEST ===")
for phase in (1, 2):
    v, t = load(phase, "val"), load(phase, "test")
    slope, intercept = np.polyfit(v.pred, v.pKd, 1)
    t_cal = slope * t.pred + intercept
    print(f"Phase {phase}: slope {slope:.2f}, intercept {intercept:.2f} | "
          f"test RMSE before {np.sqrt(((t.pred - t.pKd)**2).mean()):.3f} -> after {np.sqrt(((t_cal - t.pKd)**2).mean()):.3f}")

print("\n=== Validation curves (RMSE jumps vs Pearson stays steady?) ===")
h1 = pd.read_csv(f"{root}/results/phase1_history.csv")
h2 = pd.read_csv(f"{root}/results/phase2_history.csv")
print("Phase 1 val RMSE by epoch:", [round(x, 2) for x in h1.val_rmse])
print("Phase 1 val Pearson     :", [round(x, 2) for x in h1.val_pearson])
print("Phase 2 val RMSE by epoch:", [round(x, 2) for x in h2.val_rmse])
print("Phase 2 val Pearson     :", [round(x, 2) for x in h2.val_pearson])
print(f"Spread (std over epochs): Phase 1 RMSE {h1.val_rmse.std():.3f}, Phase 2 RMSE {h2.val_rmse.std():.3f}")
print("Phase 2 best epoch by val Pearson:", int(h2.val_pearson.idxmax()) + 1,
      "| best epoch by val RMSE:", int(h2.val_rmse.idxmin()) + 1)
