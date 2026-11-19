"""
Regenerate publication-grade `figures/downstream_sharpening.pdf` and `figures/sample_size_auc.pdf`
incorporating the verified Phase 2 [SNAP] revision experiments:
1. `downstream_sharpening.pdf`: adds the out-of-outcome ("disjoint non-outcome items") Chow-Liu
   screening curve alongside full-survey Chow-Liu, PL-AE (p=85), Passers cleaning, and Random control
   on both (a) Pennycook et al. (2020) and (b) Alvarez et al. (2019), removing the redundant suptitle.
2. `sample_size_auc.pdf`: overlays the fixed N >= 1,000 5-survey cohort alongside the all-eligible
   cohort (with eligible cohort counts n = 9, 8, 6, 5, 2 annotated cleanly on the x-axis).
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SNAP_REPO = os.environ.get("SNAP_REPO", os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..")))
sys.path.insert(0, SNAP_REPO)
os.chdir(SNAP_REPO)

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import define_necessary_elements
from dataset.loader import DataLoader
from main import prepare_for_categorical
from evaluate.detection import aligned_labels, CHECKS, _relevant
from chow_liu_rank import CLTree

PAPER_DIR = os.environ.get("PAPER_DIR", os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", "inattentiveness_paper")))
FIG_DIR = os.environ.get("FIG_DIR", os.path.join(PAPER_DIR, "figures"))


def _load_full_df(ds: str) -> pd.DataFrame:
    dc, rc, ic, adc, arc, aic = define_necessary_elements(ds, None, None, None)
    dl = DataLoader(dc, rc, ic, additional_drop_columns=adc, additional_rename_columns=arc, additional_columns_of_interest=aic)
    dl.COLUMNS_OF_INTEREST = []
    df, _ = dl.load_data(ds)
    return df.reset_index(drop=True).loc[:, lambda d: ~d.columns.duplicated()]


def _load_modeled_df(ds: str) -> pd.DataFrame:
    dc, rc, ic, adc, arc, aic = define_necessary_elements(ds, None, None, None)
    dl = DataLoader(dc, rc, ic, additional_drop_columns=adc, additional_rename_columns=arc, additional_columns_of_interest=aic)
    df, _ = dl.load_data(ds)
    return df.reset_index(drop=True)


def _ac_fail_union(ds: str, n: int) -> np.ndarray:
    lab = aligned_labels(ds)
    f = np.zeros(n, dtype=bool)
    for ch in CHECKS[ds]:
        f |= _relevant(lab, ch).astype(bool)
    return f


def _sweep_curve(effect_fn, score: np.ndarray, fracs=(0.0, 0.05, 0.10, 0.15, 0.20)):
    n = len(score)
    order = np.argsort(-score)
    return [float(effect_fn(np.setdiff1d(np.arange(n), order[: int(round(f * n))], assume_unique=False))) for f in fracs]


def _random_curve(effect_fn, n: int, fracs=(0.0, 0.05, 0.10, 0.15, 0.20), seeds=40):
    out = []
    for f in fracs:
        k = int(round(f * n))
        if k == 0:
            out.append(float(effect_fn(np.arange(n))))
            continue
        vals = []
        for s in range(seeds):
            rng = np.random.RandomState(s)
            rem = rng.choice(n, k, replace=False)
            vals.append(float(effect_fn(np.setdiff1d(np.arange(n), rem))))
        out.append(float(np.mean(vals)))
    return out


def make_downstream_figure():
    fracs = [0.0, 0.05, 0.10, 0.15, 0.20]
    pcts = [int(round(f * 100)) for f in fracs]

    # --- Panel (a): Pennycook et al. (2020) ---
    p_all = _load_full_df("pennycook_1")
    p_mod = _load_modeled_df("pennycook_1")
    n_pen = len(p_all)
    ae_pen = pd.read_csv("cache/_tuned_pennycook_1_ae_p85/errors.csv")["error"].to_numpy()
    cl_pen = (1.0 - pd.read_csv("cache/_fix9_pennycook_1_cl/errors.csv")["pct"]).to_numpy()
    acf_pen = _ac_fail_union("pennycook_1", n_pen)

    # Disjoint 62 non-headline items (drop all 30 Real1_1..15 and Fake1_1..15 headline ratings)
    hl_cols = [c for c in p_mod.columns if c.startswith("Real1_") or c.startswith("Fake1_")]
    cat_pen_disj = prepare_for_categorical(p_mod.drop(columns=hl_cols).copy()).reset_index(drop=True)
    cl_pen_disj = (1.0 - CLTree(alpha=1.0).fit(cat_pen_disj, random_state=2).score_dataframe(cat_pen_disj)["pct"]).to_numpy()

    R_pen = p_all[[c for c in p_all.columns if c.startswith("Real1_")]].apply(pd.to_numeric, errors="coerce").mean(1)
    F_pen = p_all[[c for c in p_all.columns if c.startswith("Fake1_")]].apply(pd.to_numeric, errors="coerce").mean(1)
    disc_fn = lambda idx: float(R_pen.iloc[idx].mean() - F_pen.iloc[idx].mean())

    pen_cl_in = _sweep_curve(disc_fn, cl_pen, fracs)
    pen_cl_disj = _sweep_curve(disc_fn, cl_pen_disj, fracs)
    pen_ae_in = _sweep_curve(disc_fn, ae_pen, fracs)
    pen_rnd = _random_curve(disc_fn, n_pen, fracs)
    pen_passers = disc_fn(np.where(~acf_pen)[0])

    print("Pennycook (Table 4 common scale):")
    print("  pcts       :", pcts)
    print("  CL full    :", [round(v, 4) for v in pen_cl_in])
    print("  CL disjoint:", [round(v, 4) for v in pen_cl_disj])
    print("  AE full    :", [round(v, 4) for v in pen_ae_in])
    print("  Passers    :", round(pen_passers, 4))

    # --- Panel (b): Alvarez et al. (2019) ---
    a_mod = _load_modeled_df("inattentive")
    n_alv = len(a_mod)
    ae_alv = pd.read_csv("cache/_tuned_inattentive_ae_p85/errors.csv")["error"].to_numpy()
    cl_alv = (1.0 - pd.read_csv("cache/_fix9_inattentive_cl/errors.csv")["pct"]).to_numpy()
    acf_alv = _ac_fail_union("inattentive", n_alv)

    pol_cols = [
        "support.Obamacare",
        "support.RepealDADT",
        "support.PathLegal",
        "support.CarbonLimits",
        "support.GunControl",
        "support.LimitSurveillance",
    ]
    mp = {"Support": 1.0, "Oppose": -1.0, "Indifferent": 0.0, "Don't know": 0.0}
    P_alv = a_mod[pol_cols].apply(lambda s: s.map(mp))

    def alv_fn(idx):
        c = P_alv.iloc[idx].corr().to_numpy()
        iu = np.triu_indices_from(c, 1)
        return float(np.nanmean(c[iu]))

    cat_alv_disj = prepare_for_categorical(a_mod.drop(columns=pol_cols).copy()).reset_index(drop=True)
    cl_alv_disj = (1.0 - CLTree(alpha=1.0).fit(cat_alv_disj, random_state=2).score_dataframe(cat_alv_disj)["pct"]).to_numpy()

    alv_cl_in = _sweep_curve(alv_fn, cl_alv, fracs)
    alv_cl_disj = _sweep_curve(alv_fn, cl_alv_disj, fracs)
    alv_ae_in = _sweep_curve(alv_fn, ae_alv, fracs)
    alv_rnd = _random_curve(alv_fn, n_alv, fracs)
    alv_passers = alv_fn(np.where(~acf_alv)[0])

    print("Alvarez (attitude constraint r-bar):")
    print("  CL full    :", [round(v, 4) for v in alv_cl_in])
    print("  CL disjoint:", [round(v, 4) for v in alv_cl_disj])
    print("  AE full    :", [round(v, 4) for v in alv_ae_in])
    print("  Passers    :", round(alv_passers, 4))

    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 11.0,
            "legend.fontsize": 9.3,
        }
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.15))

    # Colors: Okabe-Ito colorblind-safe palette matching existing figure
    c_cl = "#d55e00"       # vermilion (CL full)
    c_disj = "#009e73"     # bluish green (CL disjoint out-of-outcome)
    c_ae = "#0072b2"       # blue (AE full)
    c_rnd = "#8c8c8c"      # grey (random control)

    # Panel (a)
    ax1.axhline(pen_passers, color="#222222", linestyle=":", linewidth=1.35, zorder=2)
    ax1.text(19.7, pen_passers - 0.0045, "Passers (80% removed)", ha="right", va="top", fontsize=8.8, color="#333333")
    ax1.plot(pcts, pen_cl_in, "o-", color=c_cl, lw=2.1, ms=5.5, label="CL (full survey)")
    ax1.plot(pcts, pen_cl_disj, "s--", color=c_disj, lw=2.0, ms=5.2, label="CL (disjoint items)")
    ax1.plot(pcts, pen_ae_in, "^-", color=c_ae, lw=1.9, ms=5.5, label=r"$\mathrm{AE}_{85}$ (full survey)")
    ax1.plot(pcts, pen_rnd, "o--", color=c_rnd, lw=1.5, ms=4.5, alpha=0.85, label="Random (control)")
    ax1.set_title("(a) Pennycook et al. (2020): accuracy discernment", pad=8)
    ax1.set_xlabel("% of respondents removed (by score)")
    ax1.set_ylabel("True minus false accuracy gap")
    ax1.set_xticks([0, 5, 10, 15, 20])
    ax1.set_ylim(0.141, 0.243)
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="center left", bbox_to_anchor=(0.01, 0.62), frameon=False)

    # Panel (b)
    ax2.axhline(alv_passers, color="#222222", linestyle=":", linewidth=1.35, zorder=2)
    ax2.annotate(
        "Passers (64% removed)",
        xy=(18.8, alv_passers - 0.0003),
        xytext=(19.7, 0.2055),
        ha="right",
        va="center",
        fontsize=8.8,
        color="#333333",
        arrowprops=dict(arrowstyle="->", lw=0.85, color="#555555", shrinkA=1, shrinkB=2),
    )
    ax2.plot(pcts, alv_cl_in, "o-", color=c_cl, lw=2.1, ms=5.5, label="CL (full survey)")
    ax2.plot(pcts, alv_cl_disj, "s--", color=c_disj, lw=2.0, ms=5.2, label="CL (disjoint non-policy)")
    ax2.plot(pcts, alv_ae_in, "^-", color=c_ae, lw=1.9, ms=5.5, label=r"$\mathrm{AE}_{85}$ (full survey)")
    ax2.plot(pcts, alv_rnd, "o--", color=c_rnd, lw=1.5, ms=4.5, alpha=0.85, label="Random (control)")
    ax2.set_title("(b) Alvarez et al. (2019): attitude constraint", pad=8)
    ax2.set_xlabel("% of respondents removed (by score)")
    ax2.set_ylabel(r"Mean inter-item correlation ($\bar{r}$)")
    ax2.set_xticks([0, 5, 10, 15, 20])
    ax2.set_ylim(0.197, 0.249)
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper left", frameon=False)

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "downstream_sharpening.pdf")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", out_path)


def make_sample_size_figure():
    grid = [100, 250, 500, 1000, 2500]
    d = pd.read_csv("sample_size_results.csv")
    m = d.groupby(["dataset", "N"])[["AE_AUC", "CL_AUC"]].mean().reset_index()
    ds_ge1000 = [ds for ds, g in m.groupby("dataset") if g["N"].max() >= 1000]
    m_fixed5 = m[m["dataset"].isin(ds_ge1000)]

    def pct_curve(sub_df, col, max_grid=2500):
        xs, mu, cnts = [], [], []
        for n in grid:
            if n > max_grid:
                continue
            vals = []
            for _, g in sub_df.groupby("dataset"):
                fullN = g["N"].max()
                full = g.loc[g["N"] == fullN, col].iloc[0]
                if n <= fullN and np.isfinite(full) and full > 0:
                    s = g.loc[g["N"] == n, col]
                    if len(s):
                        vals.append(100.0 * s.iloc[0] / full)
            if vals:
                xs.append(n)
                mu.append(float(np.mean(vals)))
                cnts.append(len(vals))
        return np.array(xs), np.array(mu), cnts

    plt.rcParams.update({"font.size": 11.5, "axes.labelsize": 12.5, "legend.fontsize": 10.2})
    fig, ax = plt.subplots(figsize=(7.4, 4.4))

    x_cl, mu_cl, cnts = pct_curve(m, "CL_AUC")
    x_ae, mu_ae, _ = pct_curve(m, "AE_AUC")
    xf_cl, muf_cl, _ = pct_curve(m_fixed5, "CL_AUC", max_grid=1000)
    xf_ae, muf_ae, _ = pct_curve(m_fixed5, "AE_AUC", max_grid=1000)

    ax.plot(x_cl, mu_cl, "o-", ms=6.2, lw=2.3, color="#1f77b4", label="Chow-Liu (all eligible surveys)")
    ax.plot(xf_cl, muf_cl, "s--", ms=5.2, lw=1.6, color="#1f77b4", alpha=0.68, label=r"Chow-Liu (fixed $N\geq 1{,}000$ cohort, $n=5$)")
    ax.plot(x_ae, mu_ae, "o-", ms=6.2, lw=2.3, color="#d62728", label=r"$\mathrm{AE}_{85}$ (all eligible surveys)")
    ax.plot(xf_ae, muf_ae, "s--", ms=5.2, lw=1.6, color="#d62728", alpha=0.68, label=r"$\mathrm{AE}_{85}$ (fixed $N\geq 1{,}000$ cohort, $n=5$)")

    ax.axhline(100, color="grey", linestyle=":", linewidth=1.2)
    ax.axhline(98, color="grey", linestyle="--", linewidth=0.8, alpha=0.55)

    ax.set_xscale("log")
    ax.set_xticks(grid)
    ax.set_xticklabels([f"{n}\n($n={c}$)" for n, c in zip(grid, cnts)])
    ax.minorticks_off()
    ax.set_xlabel("Training sample size $N$ (eligible surveys $n$)")
    ax.set_ylabel("Detection AUC (% of full-data AUC)")
    ax.set_ylim(90, 105.5)
    ax.set_yticks([90, 92, 94, 96, 98, 100, 102, 104])
    ax.legend(loc="lower right", frameon=False, ncol=1)
    ax.grid(True, which="major", axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "sample_size_auc.pdf")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", out_path)


if __name__ == "__main__":
    make_downstream_figure()
    make_sample_size_figure()
