# MISQ Major Revision Reproducibility Scripts

This directory contains the end-to-end experiment and figure-generation scripts backing the *MIS Quarterly* Major Revision of *"Learning to Pay Attention: Unsupervised Modeling of Attentive and Inattentive Survey Respondents"* (companion PR [`ipeirotis-org/inattentiveness_paper#32`](https://github.com/ipeirotis-org/inattentiveness_paper/pull/32)), together with the verified numerical outputs under `results/`.

## Scripts & Manuscript Mapping

| Script | Output JSON / Figures | Manuscript Sections, Tables, & Figures |
|--------|-----------------------|----------------------------------------|
| `run_coherence_sim.py` | `results/coherence_sim_results.json`, `figures/coherence_simulation.pdf` | **Section 7.5, Figure 5** — 4-panel, 12-seed Monte Carlo coherence simulation comparing `Chow-Liu`, `PL-AE (p=85, m=5)`, `PCA (k=5)`, and `IRV` across common-factor loading $\lambda$, battery breadth $J$, random carelessness $\times$ reverse-keying, and 15% invariant straightlining $\times$ reverse-keying |
| `run_yrbs_revision.py` | `results/yrbs_revision_results.json` | **Tables 2–4, Sections 6.3, 7.1, 7.8, 7.9 (`kann2018`)** — 92-variable composite-disjoint YRBS refits (`{Linear, Non-Linear} x {p=100, p=85}` 5-seed ablation, 575-col one-hot MD/PCA/MCA, 28-item PC1 sign alignment, covariate-adjusted validity across 6 YRBS covariates including `qspeakenglish`, prevalence-matched binary $\widetilde{D}_M(Z)$, and 79-var drug-disjoint / 65-var substance-disjoint downstream analyses) |
| `run_all_snap_experiments.py` | `results/snap_revision_results.json` | **Tables 2, 4, S3, S12; Sections 5.2, 6.3, 7.1, 7.3, 7.7, 7.9** — 9-survey $2\times 2$ `{Linear, Non-Linear} x {p=100, p=85}` ablation (`Lin_85`), 9-survey one-hot classical baselines (10%-ridge MD, pinv MD, rank-5 MCA, rank-5/rank-$m$ PCA), 8-survey codebook-free PC1 sign alignment, 9-survey empirical coherence correlation ($\rho = +0.80$), multi-survey covariate-adjusted validity (`alvarez2019`, `mastroianni2022`), out-of-outcome downstream refits (`alvarez2019`, `pennycook2020`, `ogrady2019`), fixed-$N\ge 1{,}000$ cohort learning curves, and KerasTuner selected architectures |
| `generate_snap_figures.py` | `figures/downstream_sharpening.pdf`, `figures/sample_size_auc.pdf` | **Figure 7 (`sample_size_auc.pdf`) & Figure 9 (`downstream_sharpening.pdf`)** — fixed-$N\ge 1{,}000$ cohort learning curves and out-of-outcome ("disjoint non-outcome") downstream removal curves |
| `update_figures.py` | `figures/reconstruction_ora.pdf`, `figures/fairness_frontier.pdf` | **Figure 4 (`reconstruction_ora.pdf`) & Figure 8 (`fairness_frontier.pdf`)** — standardized `Author (Year)` dataset labels across figures |

## Usage

### 1. Standalone scripts (run directly from a clean checkout of `autoencoders_census`)

`run_coherence_sim.py` and `run_yrbs_revision.py` use only the checked-in codebase and `data/sadc_2017only_national_full.csv`:

```bash
# Controlled 4-panel coherence simulation + Figure 5 PDF
PAPER_DIR=../inattentiveness_paper python scripts/misq_revision/run_coherence_sim.py

# Full 92-variable disjoint YRBS (Kann et al., 2018) revision analyses
python scripts/misq_revision/run_yrbs_revision.py

# Standardize Author (Year) labels in Figures 4 & 8 (installs PyMuPDF if needed)
pip install -r scripts/misq_revision/requirements-revision.txt
PAPER_DIR=../inattentiveness_paper python scripts/misq_revision/update_figures.py
```

### 2. Multi-dataset scripts (run against the restored `2026-10-23` paper-data snapshot)

`run_all_snap_experiments.py` and `generate_snap_figures.py` read the eight non-YRBS benchmark CSVs, `tuned_<ds>/oracle.json` trial caches, and evaluation helpers restored from `gs://autoencoders-census-paper-data/snapshots/2026-10-23` (see PR #55 / `restore_paper_data.py` in the snapshot). By default `run_all_snap_experiments.py` recomputes all five experiment blocks from scratch (pass `--resume` with `CHECKPOINT_JSON=/path/to/ckpt.json` to resume an interrupted run):

```bash
SNAP_REPO=/path/to/restored/autoencoders_census \
  python scripts/misq_revision/run_all_snap_experiments.py

SNAP_REPO=/path/to/restored/autoencoders_census \
PAPER_DIR=../inattentiveness_paper \
  python scripts/misq_revision/generate_snap_figures.py
```
