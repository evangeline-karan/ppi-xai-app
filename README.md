# Trustworthy protein-protein binding affinity prediction with explanations

A graph neural network that predicts the binding affinity (pKd = -log10 Kd) of a protein-protein complex, including
the effect of point mutations, and explains each prediction with residue-level and residue-pair-level evidence.
This repository contains the training code, the saved model and a Streamlit app.

> Research prototype. Not for clinical or safety-critical use.

## Method in brief

1. **Data**: PPB-Affinity, 12,062 samples from 3,032 PDB entries. After checking every sample against its 3D
   structure (chains present, mutated residue found and matching the wild-type letter), 11,703 samples from
   3,019 PDB entries remain. Splits are grouped by PDB ID (train 9,355 / validation 1,138 / test 1,210).
2. **Features**: ESM-2 (650M, 1280-d) residue embeddings, amino-acid one-hot, physicochemical properties and a
   mutation block (flag, wild-type and mutant identity, property change). Residues within 8 A (C-alpha) are connected.
3. **Models**: Phase 1 = two independent GATv2 encoders, mean pooling, MLP (baseline). Phase 2 = the same encoders plus a
   sparse ligand-receptor residue-pair matrix (pairs under 15 A), a contact auxiliary loss and attention pooling.
4. **Explanations**: residue occlusion, Integrated Gradients and the pair attention weights.
5. **Calibration**: a linear correction fitted on the validation set only (slope 0.61).

## Results (held-out test set)

| Model | RMSE | MAE | Pearson | Spearman | R2 | Within-complex Spearman |
|---|---|---|---|---|---|---|
| Predict the training mean | 1.85 | | | | | |
| Phase 1 (calibrated) | 1.445 | 1.160 | 0.680 | 0.680 | 0.390 | 0.098 |
| Phase 2 (calibrated) | 1.428 | 1.122 | 0.680 | 0.672 | 0.405 | 0.165 |

Strong (pKd >= 7) vs weak binder AUROC: 0.866 (Phase 1), 0.858 (Phase 2).

Explanation checks on 60 held-out complexes (Phase 2):

| Check | Result |
|---|---|
| Structural agreement, AUROC for finding interface residues (0.5 = chance) | IG 0.60, occlusion 0.63, pair attention 0.66 |
| Faithfulness, mean change in pKd when the top-5 residues are masked | 0.30 (IG), 0.18 (occlusion), 0.012 (random) |
| Stability of IG under small noise (Spearman) | 0.97 |
| IG completeness error | 1.8% |

**Limitations.** Phase 1 and Phase 2 are not distinguishable with a single run. Ranking mutants of the same
complex is weak. Explanations are faithful to the model but only modestly agree with the structural interface.
There is no uncertainty estimate yet. Results are from one run on one dataset.

## Repository layout

```
app.py                   Streamlit app
src/                     model, features, structure helpers, plots
app_assets/              trained weights, 60 demo explanations, results summary
training/                the Colab cells used for training, evaluation and explanation (run in this order:
                         phase1..., phase2..., calibrate..., diagnose..., phase3_xai, export_app_assets)
requirements.txt         demo mode (light)
requirements-live.txt    live mode (adds torch, torch_geometric, biopython, fair-esm)
```

The data preparation steps (cleaning, PDB-grouped split, structure download and validation, ESM-2 embeddings) are
not yet saved as a script in `training/`.

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

Demo mode works with this light install. Live mode (predict your own complex) needs `requirements-live.txt`,
about 4 GB of RAM and the environment variable `PPI_LIVE=1`:

```
pip install -r requirements-live.txt
PPI_LIVE=1 streamlit run app.py
```

## Deploy

**Streamlit Community Cloud (demo mode).** Push this repository to GitHub, then at share.streamlit.io choose
"New app", select the repository and `app.py`, and deploy.

**Hugging Face Space (demo + live mode).** Create a Space with the Streamlit SDK, add the files of this repository,
replace `requirements.txt` with the content of `requirements-live.txt`, and add a Space variable `PPI_LIVE` = `1`.
The first prediction downloads ESM-2 (about 2.5 GB).

## Live mode notes

- Mutations use PDB residue numbers, written `Chain_WildTypePositionMutant`, for example `I_L38G, I_R46A`.
- The wild-type letter is checked against the structure; mismatches are reported instead of guessed.
- Live mode is limited to 1,500 residues per complex.
