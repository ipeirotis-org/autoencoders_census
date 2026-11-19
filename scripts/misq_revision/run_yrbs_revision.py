"""
Comprehensive YRBS (Kann et al., 2018 / sadc_2017) Reanalysis for MISQ Revision.

Covers:
  - Major 2.4c: 92-variable disjoint primary fit (excluding all 6 mischievous-composite
    items) vs. 98-variable historical fit and 85-variable weight/BMI-excluded fit
  - Major 2.1b & 2.1c: Full {Non-Linear, Zero-Layer Linear} x {p=100, p=85} 5-seed
    ablation under config/simple_autoencoder.yaml (128-unit, m=2, 5 epochs, B=256,
    lr=1e-4) + linear m=8 check + 5-fold CV & 3,000-draw bootstrap 95% CIs
  - Major 2.2a & 2.2b: Full-feature one-hot regularized Mahalanobis (MD_1hot) and
    rank-5 PCA / MCA reconstruction error on exact autoencoder input matrix, plus
    codebook-free PC1 sign-aligned PT and EO on ordinal items
  - Major 2.4b: Covariate-adjusted validity Corr(M, C | Z) and AUC(M, C | Z) across
    age, grade, race, sex, sexual_identity, and how_well_speak_English (individually
    and jointly)
  - Minor 3.4: Continuous passer D_M(Z) and prevalence-matched binary D_tilde_M(Z)
  - Major 2.3: Out-of-outcome ("drug-disjoint" 79-var and "substance-disjoint" 65-var)
    downstream LGBQ-heterosexual drug-use prevalence disparity analysis
"""
import hashlib
import json
import os
import re
import sys
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("REPO_ROOT", os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..")))
sys.path.insert(0, REPO)
os.chdir(REPO)

from dataset.loader import DataLoader
from features.transform import Table2Vector
from utils import define_necessary_elements
from chow_liu_rank import CLTree

OUT_JSON = os.environ.get("OUT_JSON", os.path.join(SCRIPT_DIR, "results", "yrbs_revision_results.json"))

COMP_6 = [
    "carrot_eating", "green _salad_eating", "fruit_juice_drinking",
    "fruit_eating", "stheight_cat", "stweight_cat",
]
WEIGHT_BMI_7 = [
    "obese", "overweight", "perception_of_weight", "weight_loss",
    "weight_cat", "bmi_cat", "bmipct_cat",
]
DRUG_13 = [
    "ever_marijuana_use", "initiation_of_marijuana_use", "current_marijuana_use",
    "ever_cocaine_use", "ever_inhalant_use", "ever_heroin_use",
    "ever_methamphetamine_use", "ever_ecstasy_use", "ever_synthetic_marijuana_use",
    "ever_steroid_use", "illegal_injected_drug_use", "illegal_drugs_at_school",
    "ever_used_LSD",
]
ALC_TOB_14 = [
    "ever_cigarette_use", "current_cigarette_use", "smoking_amounts_per_day",
    "electronic_vapor_product_use", "current_electronic_vapor_product_use",
    "current_cigar_use", "ever_alcohol_use", "initiation_of_alcohol_use",
    "current_alcohol_use", "source_of_alcohol", "largest_number_of_drinks",
    "drinking_and_driving", "riding_with_a_drinking_driver", "alcohol/drugs_at_sex",
]


def vectorize_df(df_sub, vtypes_all):
    vtypes = {c: vtypes_all[c] for c in df_sub.columns}
    vec = Table2Vector(vtypes)
    X_df = vec.vectorize_table(df_sub)
    X_oh = X_df.values.astype(np.float32)
    cards = [int(df_sub[c].nunique(dropna=False)) for c in df_sub.columns]
    assert sum(cards) == X_oh.shape[1], (sum(cards), X_oh.shape[1])
    return X_oh, cards


class BlockSoftmaxAE:
    """Fast vectorized per-variable softmax autoencoder (non-linear or zero-layer linear)."""

    def __init__(self, cards, arch="nonlinear", hidden=128, bottleneck=2, lr=1e-4, p=85.0):
        self.cards = list(cards)
        self.V = len(self.cards)
        self.D = int(sum(self.cards))
        self.arch = arch
        self.hidden = hidden
        self.bottleneck = bottleneck
        self.lr = lr
        self.p = p

        var_ids = []
        for v, k in enumerate(self.cards):
            var_ids.extend([v] * k)
        self.var_ids = np.array(var_ids, dtype=np.int32)
        self.var_ids_tf = tf.constant(self.var_ids, dtype=tf.int32)

        B_mat = np.zeros((self.D, self.V), dtype=np.float32)
        B_mat[np.arange(self.D), self.var_ids] = 1.0
        self.B_mat_tf = tf.constant(B_mat, dtype=tf.float32)
        self.log_cards = np.maximum(np.log(np.array(self.cards, dtype=np.float32)), 1e-10)
        self.log_cards_tf = tf.constant(self.log_cards[None, :], dtype=tf.float32)
        self._build()

    def _build(self):
        inp = keras.Input(shape=(self.D,))
        if self.arch == "nonlinear":
            h = keras.layers.Dense(self.hidden, activation="relu", name="enc_h")(inp)
            z = keras.layers.Dense(self.bottleneck, activation="relu", name="lat_z")(h)
            dec = keras.layers.Dense(self.hidden, activation="relu", name="dec_h")(z)
            logits = keras.layers.Dense(self.D, activation=None, name="out_logits")(dec)
        elif self.arch == "linear":
            z = keras.layers.Dense(self.bottleneck, activation=None, name="lat_z")(inp)
            logits = keras.layers.Dense(self.D, activation=None, name="out_logits")(z)
        else:
            raise ValueError(self.arch)

        self.model = keras.Model(inp, logits)
        self.opt = keras.optimizers.Adam(learning_rate=self.lr)
        dummy = tf.zeros((4, self.D), dtype=tf.float32)
        with tf.GradientTape() as tape:
            l0 = tf.reduce_mean(self.model(dummy, training=True))
        g0 = tape.gradient(l0, self.model.trainable_variables)
        self.opt.apply_gradients(zip(g0, self.model.trainable_variables))

        @tf.function
        def sample_losses(x_batch, training):
            logits_b = self.model(x_batch, training=training)
            # Per-variable log-softmax via transposed unsorted_segment_max + block sum
            max_v = tf.transpose(
                tf.math.unsorted_segment_max(tf.transpose(logits_b), self.var_ids_tf, self.V)
            )
            shifted = logits_b - tf.gather(max_v, self.var_ids_tf, axis=1)
            sum_exp_v = tf.matmul(tf.exp(shifted), self.B_mat_tf)
            log_probs = shifted - tf.gather(tf.math.log(sum_exp_v + 1e-12), self.var_ids_tf, axis=1)
            ce_v = -tf.matmul(x_batch * log_probs, self.B_mat_tf) / self.log_cards_tf
            return tf.reduce_mean(ce_v, axis=-1)

        @tf.function
        def train_step(x_batch):
            with tf.GradientTape() as tape:
                sl = sample_losses(x_batch, True)
                if self.p >= 100.0:
                    loss = tf.reduce_mean(sl)
                else:
                    bsz = tf.shape(sl)[0]
                    k_keep = tf.maximum(
                        1, tf.cast(tf.math.floor(tf.cast(bsz, tf.float32) * (self.p / 100.0)), tf.int32)
                    )
                    sorted_sl = tf.sort(sl, direction="ASCENDING")
                    loss = tf.reduce_mean(sorted_sl[:k_keep])
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
            return loss

        self._sample_losses = sample_losses
        self._train_step = train_step

    def reset_weights(self, seed):
        rng = np.random.default_rng(seed)
        for layer in self.model.layers:
            if hasattr(layer, "kernel") and layer.kernel is not None:
                fin, fout = layer.kernel.shape
                if layer.name == "out_logits":
                    # Blockwise GlorotUniform matching separate Dense(K_v) heads
                    w = np.zeros((fin, fout), dtype=np.float32)
                    offset = 0
                    for kv in self.cards:
                        lim = np.sqrt(6.0 / (fin + kv))
                        w[:, offset:offset + kv] = rng.uniform(-lim, lim, size=(fin, kv)).astype(np.float32)
                        offset += kv
                else:
                    lim = np.sqrt(6.0 / (fin + fout))
                    w = rng.uniform(-lim, lim, size=(fin, fout)).astype(np.float32)
                layer.kernel.assign(w)
            if hasattr(layer, "bias") and layer.bias is not None:
                layer.bias.assign(np.zeros(layer.bias.shape, dtype=np.float32))
        for var in self.opt.variables:
            var.assign(tf.zeros_like(var))

    def fit_and_score(self, X_oh, seed, epochs=5, batch_size=256, train_mask=None):
        self.reset_weights(seed)
        rng = np.random.default_rng(seed + 5000)
        X_train = X_oh[train_mask] if train_mask is not None else X_oh
        n_tr = X_train.shape[0]
        for _ in range(epochs):
            perm = rng.permutation(n_tr)
            X_shuf = X_train[perm]
            for start in range(0, n_tr, batch_size):
                self._train_step(tf.constant(X_shuf[start:start + batch_size]))
        # Score full dataset in chunks
        scores = np.zeros(X_oh.shape[0], dtype=np.float32)
        for start in range(0, X_oh.shape[0], 2048):
            end = min(X_oh.shape[0], start + 2048)
            scores[start:end] = self._sample_losses(tf.constant(X_oh[start:end]), False).numpy()
        return scores


def bootstrap_auc_ci(y, score, n_boot=3000, seed=42):
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)
    # Stratified bootstrap preserves positive/negative counts cleanly
    aucs = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        bp = rng.choice(pos_idx, size=n_pos, replace=True)
        bn = rng.choice(neg_idx, size=n_neg, replace=True)
        idx = np.concatenate([bp, bn])
        yb = np.concatenate([np.ones(n_pos, dtype=int), np.zeros(n_neg, dtype=int)])
        aucs[b] = roc_auc_score(yb, score[idx])
    return [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]


def full_feature_baselines(X_oh, y):
    # 1. Standardized pseudo-inverse Mahalanobis distance on one-hot features
    mu = X_oh.mean(axis=0, keepdims=True)
    sd = X_oh.std(axis=0, ddof=1, keepdims=True)
    Z_oh = (X_oh - mu) / np.where(sd > 1e-8, sd, 1.0)
    cov = np.cov(Z_oh, rowvar=False)
    pinv = np.linalg.pinv(cov, rcond=1e-5)
    md_pinv = np.einsum("ij,jk,ik->i", Z_oh, pinv, Z_oh)

    # 2. Ridge-regularized covariance Mahalanobis (shrinkage alpha=0.10 toward identity)
    d = cov.shape[0]
    cov_ridge = 0.90 * cov + 0.10 * np.eye(d)
    inv_ridge = np.linalg.inv(cov_ridge)
    md_ridge = np.einsum("ij,jk,ik->i", Z_oh, inv_ridge, Z_oh)

    # 3. Rank-k PCA squared reconstruction error on centered one-hot features
    X_c = X_oh - mu
    pca_res = {}
    for k in [2, 5, 10]:
        pca = PCA(n_components=k, svd_solver="randomized", random_state=42)
        proj = pca.fit_transform(X_c)
        recon = pca.inverse_transform(proj)
        err = np.sum((X_c - recon) ** 2, axis=1)
        pca_res[f"pca_k{k}_auc"] = float(roc_auc_score(y, err))

    # 4. MCA / Chi-square standardized one-hot rank-5 reconstruction error
    p_col = np.clip(X_oh.mean(axis=0, keepdims=True), 1e-6, 1.0)
    X_mca = (X_oh - p_col) / np.sqrt(p_col)
    pca_mca = PCA(n_components=5, svd_solver="randomized", random_state=42)
    recon_mca = pca_mca.inverse_transform(pca_mca.fit_transform(X_mca))
    err_mca = np.sum((X_mca - recon_mca) ** 2, axis=1)

    return {
        "md_1hot_pinv_auc": float(roc_auc_score(y, md_pinv)),
        "md_1hot_ridge_auc": float(roc_auc_score(y, md_ridge)),
        "mca_k5_auc": float(roc_auc_score(y, err_mca)),
        **pca_res,
    }


def psychometric_baselines_with_pc1(df_sub, y):
    # Select non-demographic ordinal columns whose non-missing values start with digits
    ord_cols = []
    for c in df_sub.columns:
        if c in ("age", "grade"):
            continue
        u = [v for v in df_sub[c].unique() if v != "missing"]
        if len(u) >= 3 and all(re.match(r"^\d+", str(v)) for v in u):
            ord_cols.append(c)

    N, J = len(df_sub), len(ord_cols)
    M = np.full((N, J), np.nan)
    Ks = np.zeros(J)
    for j, c in enumerate(ord_cols):
        u = sorted([v for v in df_sub[c].unique() if v != "missing"],
                   key=lambda s: float(re.match(r"^\d+", str(s)).group(0)))
        Ks[j] = len(u)
        mp = {v: idx + 1.0 for idx, v in enumerate(u)}
        M[:, j] = [mp.get(val, np.nan) for val in df_sub[c].values]

    # Longstring & IRV (never sign-flipped!)
    ls = np.zeros(N)
    for i in range(N):
        best, cur, prev = 0, 0, None
        for val in M[i]:
            if np.isnan(val):
                cur, prev = 0, None
            elif val == prev:
                cur += 1
                if cur > best:
                    best = cur
            else:
                cur, prev = 1, val
                if cur > best:
                    best = cur
        ls[i] = best
    irv = -np.nanstd(M, axis=1, ddof=1)

    col_means = np.nanmean(M, axis=0)
    col_stds = np.nanstd(M, axis=0, ddof=1)
    M_imp = np.where(np.isnan(M), col_means, M)
    M_z = (M_imp - col_means) / np.where(col_stds > 1e-8, col_stds, 1.0)

    # Mahalanobis on standardized ordinal items
    cov = np.cov(M_z, rowvar=False)
    md = np.einsum("ij,jk,ik->i", M_z, np.linalg.pinv(cov), M_z)

    def compute_pt_eo(mat):
        means_j = np.nanmean(mat, axis=0)
        pt_s = np.zeros(N)
        eo_s = np.zeros(N)
        even_idx = np.arange(0, J, 2)
        odd_idx = np.arange(1, J, 2)
        # Split into item pairs or even/odd profile correlation / half means
        for i in range(N):
            ok = ~np.isnan(mat[i])
            if ok.sum() >= 2 and np.std(mat[i, ok]) > 1e-12:
                pt_s[i] = -np.corrcoef(mat[i, ok], means_j[ok])[0, 1]
            # Even-odd: correlate even and odd item z-scores (or paired items up to min len)
            m_len = min(len(even_idx), len(odd_idx))
            ev = mat[i, even_idx[:m_len]]
            od = mat[i, odd_idx[:m_len]]
            ok_eo = (~np.isnan(ev)) & (~np.isnan(od))
            if ok_eo.sum() >= 2 and np.std(ev[ok_eo]) > 1e-12 and np.std(od[ok_eo]) > 1e-12:
                r_eo = np.corrcoef(ev[ok_eo], od[ok_eo])[0, 1]
                sb = (2.0 * r_eo) / (1.0 + abs(r_eo)) if (1.0 + abs(r_eo)) > 1e-8 else 0.0
                eo_s[i] = -sb
        return pt_s, eo_s

    pt_raw, eo_raw = compute_pt_eo(M)

    # Codebook-free PC1 sign alignment on standardized items
    pca1 = PCA(n_components=1, random_state=42).fit(M_z)
    loadings = pca1.components_[0]
    # Orient PC1 so majority of items have positive loading
    if np.sum(loadings >= 0) < len(loadings) / 2.0:
        loadings = -loadings
    neg_items = [ord_cols[j] for j in range(J) if loadings[j] < 0]
    M_aligned = M.copy()
    for j in range(J):
        if loadings[j] < 0:
            M_aligned[:, j] = (Ks[j] + 1.0) - M[:, j]
    pt_pc1, eo_pc1 = compute_pt_eo(M_aligned)
    # Also check standardized M_z PC1-aligned EO/PT
    M_z_aligned = M_z * np.where(loadings < 0, -1.0, 1.0)[None, :]
    pt_z_raw, eo_z_raw = compute_pt_eo(M_z)
    pt_z_pc1, eo_z_pc1 = compute_pt_eo(M_z_aligned)

    return {
        "n_ordinal_items": J,
        "n_pc1_flipped": len(neg_items),
        "pc1_flipped_items": neg_items,
        "ls_auc": float(roc_auc_score(y, ls)),
        "irv_auc": float(roc_auc_score(y, irv)),
        "md_ord_auc": float(roc_auc_score(y, md)),
        "pt_raw_auc": float(roc_auc_score(y, pt_raw)),
        "pt_pc1_auc": float(roc_auc_score(y, pt_pc1)),
        "eo_raw_auc": float(roc_auc_score(y, eo_raw)),
        "eo_pc1_auc": float(roc_auc_score(y, eo_pc1)),
        "pt_z_raw_auc": float(roc_auc_score(y, pt_z_raw)),
        "pt_z_pc1_auc": float(roc_auc_score(y, pt_z_pc1)),
        "eo_z_raw_auc": float(roc_auc_score(y, eo_z_raw)),
        "eo_z_pc1_auc": float(roc_auc_score(y, eo_z_pc1)),
    }


def correlation_ratio(values, groups):
    """Eta(values, groups): sqrt(SS_between / SS_total)."""
    v = np.asarray(values, dtype=float)
    g = np.asarray(groups)
    grand = v.mean()
    ss_tot = np.sum((v - grand) ** 2)
    if ss_tot <= 1e-15:
        return 0.0
    ss_bet = 0.0
    for lvl in pd.unique(g):
        mask = (g == lvl)
        if mask.sum() > 0:
            ss_bet += mask.sum() * ((v[mask].mean() - grand) ** 2)
    return float(np.sqrt(max(0.0, ss_bet / ss_tot)))


def covariate_adjusted_validity(score, y, df_full):
    axes = ["age", "grade", "race", "sex", "sexual_identity", "how_well_speak_English"]
    out = {}
    y_f = y.astype(float)
    out["unadjusted_corr"] = float(np.corrcoef(score, y_f)[0, 1])
    out["unadjusted_auc"] = float(roc_auc_score(y, score))

    for ax in axes:
        g = df_full[ax].astype(str).values
        # Exclude 'missing' or include as stratum; check non-missing first and all strata
        valid = (g != "missing")
        Z_dum = pd.get_dummies(g[valid], drop_first=True, dtype=float).values
        Z_design = np.column_stack([np.ones(valid.sum()), Z_dum])
        s_sub = score[valid]
        y_sub = y_f[valid]
        s_res = s_sub - Z_design @ np.linalg.lstsq(Z_design, s_sub, rcond=None)[0]
        y_res = y_sub - Z_design @ np.linalg.lstsq(Z_design, y_sub, rcond=None)[0]
        pcorr = float(np.corrcoef(s_res, y_res)[0, 1])

        # Within-stratum pair-weighted AUC
        w_sum, auc_sum = 0.0, 0.0
        for lvl in np.unique(g[valid]):
            m = valid & (g == lvl)
            n1 = int(y[m].sum())
            n0 = int((1 - y[m]).sum())
            if n1 > 0 and n0 > 0:
                w = float(n1 * n0)
                auc_sum += w * roc_auc_score(y[m], score[m])
                w_sum += w
        strat_auc = float(auc_sum / w_sum) if w_sum > 0 else None
        res_auc = float(roc_auc_score(y[valid], s_res))
        out[ax] = {
            "n_nonmissing": int(valid.sum()),
            "partial_corr": pcorr,
            "stratum_weighted_auc": strat_auc,
            "residualized_auc": res_auc,
        }

    # Joint adjustment across all 6 covariates simultaneously (including 'missing' category so N=14,765)
    Z_all = pd.get_dummies(df_full[axes].astype(str), drop_first=True, dtype=float).values
    Z_all_des = np.column_stack([np.ones(len(df_full)), Z_all])
    s_res_all = score - Z_all_des @ np.linalg.lstsq(Z_all_des, score, rcond=None)[0]
    y_res_all = y_f - Z_all_des @ np.linalg.lstsq(Z_all_des, y_f, rcond=None)[0]
    out["joint_all_6"] = {
        "n": int(len(df_full)),
        "partial_corr": float(np.corrcoef(s_res_all, y_res_all)[0, 1]),
        "residualized_auc": float(roc_auc_score(y, s_res_all)),
    }
    return out


def fairness_metrics(score, y, df_full):
    axes = ["sex", "age", "grade", "race", "sexual_identity", "how_well_speak_English"]
    h = int(y.sum())
    top_h = np.zeros(len(y), dtype=float)
    top_h[np.argsort(-score)[:h]] = 1.0
    pass_mask = (y == 0)
    res = {}
    for ax in axes:
        g_all = df_full[ax].astype(str).values
        nm_all = (g_all != "missing")
        nm_pass = pass_mask & nm_all
        dm_pass = correlation_ratio(score[nm_pass], g_all[nm_pass])
        dtilde_full = correlation_ratio(top_h[nm_all], g_all[nm_all])
        dc_full = correlation_ratio(y[nm_all].astype(float), g_all[nm_all])
        res[ax] = {
            "D_M_passers_cont": dm_pass,
            "D_tilde_M_full_bin": dtilde_full,
            "D_C_full_bin": dc_full,
        }
    return res


def drug_disparity_eval(raw_df, df_full, y, scores_dict):
    m_lgbq = df_full["sexual_identity"].isin(["Gay or lesbian", "Bisexual", "Not sure"]).values
    m_het = (df_full["sexual_identity"] == "Heterosexual (straight)").values
    keep_pass = (y == 0)
    h = int(y.sum())  # 230

    sets = {
        "qn48_to_qn58_11items": [f"qn{k}" for k in range(48, 59)],
        "qn46_to_qn58_plus_lsd_14items": [f"qn{k}" for k in range(46, 59)] + ["qnhallucdrug"],
    }
    out = {}
    for sname, cols in sets.items():
        full_gaps, pass_gaps = [], []
        for c in cols:
            v = raw_df[c].values
            val = (v == 1) | (v == 2)
            pos = (v == 1).astype(float)
            full_gaps.append((pos[val & m_lgbq].mean() - pos[val & m_het].mean()) * 100.0)
            pass_gaps.append((pos[val & keep_pass & m_lgbq].mean() - pos[val & keep_pass & m_het].mean()) * 100.0)

        s_entry = {
            "full_pp": float(np.mean(full_gaps)),
            "passers_pp": float(np.mean(pass_gaps)),
            "detectors": {},
        }
        for dname, sc in scores_dict.items():
            flagged = np.zeros(len(y), dtype=bool)
            flagged[np.argsort(-sc)[:h]] = True
            keep = ~flagged
            det_gaps, pct_reds = [], []
            for idx_c, c in enumerate(cols):
                v = raw_df[c].values
                val = (v == 1) | (v == 2)
                pos = (v == 1).astype(float)
                gd = (pos[val & keep & m_lgbq].mean() - pos[val & keep & m_het].mean()) * 100.0
                det_gaps.append(gd)
                pct_reds.append((full_gaps[idx_c] - gd) / full_gaps[idx_c] * 100.0)
            r_lgbq = float(flagged[m_lgbq].mean())
            r_het = float(flagged[m_het].mean())
            s_entry["detectors"][dname] = {
                "post_pp": float(np.mean(det_gaps)),
                "mean_pct_reduction": float(np.mean(pct_reds)),
                "lgbq_flag_pct": r_lgbq * 100.0,
                "het_flag_pct": r_het * 100.0,
                "lgbq_het_flag_ratio": float(r_lgbq / r_het) if r_het > 0 else None,
            }
        out[sname] = s_entry
    return out


def main():
    t0 = time.time()
    csv_path = os.path.join(REPO, "data/sadc_2017only_national_full.csv")
    sha = hashlib.sha256(open(csv_path, "rb").read()).hexdigest()
    raw_df = pd.read_csv(csv_path, low_memory=False)

    drop_cols, rename_cols, interest_cols, add_drop, add_rename, add_int = define_necessary_elements(
        "sadc_2017", None, None, None
    )
    dl = DataLoader(
        drop_cols, rename_cols, interest_cols,
        additional_drop_columns=add_drop,
        additional_rename_columns=add_rename,
        additional_columns_of_interest=add_int,
        max_unique_values=20,
    )
    df98, meta = dl.load_data("sadc_2017")
    vtypes_all = meta["variable_types"]
    y = dl.find_outlier_data_sadc_2017("sadc_2017", ["outlier"])["outlier"].astype(int).values

    df92 = df98.drop(columns=COMP_6)
    df85 = df92.drop(columns=WEIGHT_BMI_7)
    df79_nodrugs = df92.drop(columns=DRUG_13)
    df65_nosubstance = df79_nodrugs.drop(columns=ALC_TOB_14)

    X98, cards98 = vectorize_df(df98, vtypes_all)
    X92, cards92 = vectorize_df(df92, vtypes_all)
    X79, cards79 = vectorize_df(df79_nodrugs, vtypes_all)
    X65, cards65 = vectorize_df(df65_nosubstance, vtypes_all)

    res = {
        "provenance": {
            "csv_sha256": sha,
            "N": int(len(df98)),
            "positives_h": int(y.sum()),
            "shapes": {
                "df98": {"V": int(df98.shape[1]), "D": int(X98.shape[1]), "AFV": float(X98.shape[1] / df98.shape[1])},
                "df92_disjoint": {"V": int(df92.shape[1]), "D": int(X92.shape[1]), "AFV": float(X92.shape[1] / df92.shape[1])},
                "df85_nobmi": {"V": int(df85.shape[1])},
                "df79_nodrugs": {"V": int(df79_nodrugs.shape[1]), "D": int(X79.shape[1])},
                "df65_nosubstance": {"V": int(df65_nosubstance.shape[1]), "D": int(X65.shape[1])},
            },
        }
    }
    print("Shapes:", json.dumps(res["provenance"]["shapes"], indent=2))

    # 1. Chow-Liu across 98, 92, 85, 79, 65 variable sets + 5-fold CV on 92 & 98
    print("Fitting Chow-Liu trees...")
    cl98 = CLTree(alpha=1.0).fit(df98, mi_subsample=None)
    s_cl98 = -cl98.log_likelihood(df98)

    cl92 = CLTree(alpha=1.0).fit(df92, mi_subsample=None)
    s_cl92 = -cl92.log_likelihood(df92)

    cl85 = CLTree(alpha=1.0).fit(df85, mi_subsample=None)
    s_cl85 = -cl85.log_likelihood(df85)

    cl79 = CLTree(alpha=1.0).fit(df79_nodrugs, mi_subsample=None)
    s_cl79 = -cl79.log_likelihood(df79_nodrugs)

    cl65 = CLTree(alpha=1.0).fit(df65_nosubstance, mi_subsample=None)
    s_cl65 = -cl65.log_likelihood(df65_nosubstance)

    # 5-fold StratifiedKFold CV for CL92 and CL98
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_cl92 = np.zeros(len(df92))
    oof_cl98 = np.zeros(len(df98))
    fold_aucs_92, fold_aucs_98 = [], []
    for tr_idx, te_idx in skf.split(df92, y):
        m92 = CLTree(alpha=1.0).fit(df92.iloc[tr_idx], mi_subsample=None)
        oof_cl92[te_idx] = -m92.log_likelihood(df92.iloc[te_idx])
        fold_aucs_92.append(float(roc_auc_score(y[te_idx], oof_cl92[te_idx])))

        m98 = CLTree(alpha=1.0).fit(df98.iloc[tr_idx], mi_subsample=None)
        oof_cl98[te_idx] = -m98.log_likelihood(df98.iloc[te_idx])
        fold_aucs_98.append(float(roc_auc_score(y[te_idx], oof_cl98[te_idx])))

    res["chow_liu"] = {
        "cl98_full_auc": float(roc_auc_score(y, s_cl98)),
        "cl98_5fold_mean_auc": float(np.mean(fold_aucs_98)),
        "cl98_ci95": bootstrap_auc_ci(y, s_cl98),
        "cl98_corr_MC": float(np.corrcoef(s_cl98, y)[0, 1]),
        "cl92_full_auc": float(roc_auc_score(y, s_cl92)),
        "cl92_5fold_mean_auc": float(np.mean(fold_aucs_92)),
        "cl92_ci95": bootstrap_auc_ci(y, s_cl92),
        "cl92_corr_MC": float(np.corrcoef(s_cl92, y)[0, 1]),
        "cl85_full_auc": float(roc_auc_score(y, s_cl85)),
        "cl79_nodrugs_auc": float(roc_auc_score(y, s_cl79)),
        "cl65_nosubstance_auc": float(roc_auc_score(y, s_cl65)),
    }
    print("Chow-Liu:", json.dumps(res["chow_liu"], indent=2))

    # 2. Full 2x2 Autoencoder ablation on 92-var disjoint (and 98-var reference) over 5 seeds
    print("Running 5-seed 2x2 Autoencoder ablation...")
    seeds = [1, 2, 3, 4, 5]
    ae_ablation = {"df92": {}, "df98": {}}
    median_scores = {}

    for dtag, X_mat, cards_vec in [("df92", X92, cards92), ("df98", X98, cards98)]:
        configs = [
            ("nonlinear_p85", "nonlinear", 2, 85.0),
            ("nonlinear_p100", "nonlinear", 2, 100.0),
            ("linear_m2_p85", "linear", 2, 85.0),
            ("linear_m2_p100", "linear", 2, 100.0),
            ("linear_m8_p85", "linear", 8, 85.0),
            ("linear_m8_p100", "linear", 8, 100.0),
        ]
        for cname, arch, bneck, pval in configs:
            trainer = BlockSoftmaxAE(cards_vec, arch=arch, hidden=128, bottleneck=bneck, lr=1e-4, p=pval)
            seed_aucs, seed_scores = [], []
            for sd in seeds:
                # Match 80% train split as in simple_autoencoder.yaml (test_size=0.2)
                rng_sp = np.random.default_rng(100 + sd)
                tr_mask = rng_sp.random(X_mat.shape[0]) >= 0.2
                sc = trainer.fit_and_score(X_mat, seed=1000 + sd, epochs=5, batch_size=256, train_mask=tr_mask)
                seed_aucs.append(float(roc_auc_score(y, sc)))
                seed_scores.append(sc)
            order = np.argsort(seed_aucs)
            med_idx = int(order[len(seeds) // 2])
            med_sc = seed_scores[med_idx]
            median_scores[f"{dtag}_{cname}"] = med_sc
            ae_ablation[dtag][cname] = {
                "seed_aucs": seed_aucs,
                "median_auc": float(np.median(seed_aucs)),
                "mean_auc": float(np.mean(seed_aucs)),
                "iqr_auc": float(np.percentile(seed_aucs, 75) - np.percentile(seed_aucs, 25)),
                "min_auc": float(np.min(seed_aucs)),
                "max_auc": float(np.max(seed_aucs)),
                "median_seed_ci95": bootstrap_auc_ci(y, med_sc) if cname == "nonlinear_p85" else None,
                "median_seed_corr_MC": float(np.corrcoef(med_sc, y)[0, 1]),
            }
            print(f"  {dtag} {cname:18s}: median={np.median(seed_aucs):.4f} seeds={[round(a,4) for a in seed_aucs]}")

    # Also fit 5-seed nonlinear_p85 on df79_nodrugs and df65_nosubstance for out-of-outcome check!
    for dtag, X_mat, cards_vec in [("df79_nodrugs", X79, cards79), ("df65_nosubstance", X65, cards65)]:
        tr_nd = BlockSoftmaxAE(cards_vec, arch="nonlinear", hidden=128, bottleneck=2, lr=1e-4, p=85.0)
        nd_aucs, nd_scores = [], []
        for sd in seeds:
            rng_sp = np.random.default_rng(100 + sd)
            tr_mask = rng_sp.random(X_mat.shape[0]) >= 0.2
            sc = tr_nd.fit_and_score(X_mat, seed=1000 + sd, epochs=5, batch_size=256, train_mask=tr_mask)
            nd_aucs.append(float(roc_auc_score(y, sc)))
            nd_scores.append(sc)
        med_idx = int(np.argsort(nd_aucs)[len(seeds) // 2])
        median_scores[f"{dtag}_nonlinear_p85"] = nd_scores[med_idx]
        ae_ablation[dtag] = {
            "nonlinear_p85_median_auc": float(np.median(nd_aucs)),
            "seed_aucs": nd_aucs,
        }
        print(f"  {dtag} nonlinear_p85: median={np.median(nd_aucs):.4f}")

    res["ae_ablation"] = ae_ablation
    res["corr_AE_CL_df92"] = float(np.corrcoef(median_scores["df92_nonlinear_p85"], s_cl92)[0, 1])

    # 3. Full-feature classical baselines & PC1-aligned psychometric baselines
    print("Computing full-feature & PC1-aligned psychometric baselines...")
    res["full_feature_baselines"] = {
        "df92": full_feature_baselines(X92, y),
        "df98": full_feature_baselines(X98, y),
    }
    res["psychometric_baselines"] = {
        "df92": psychometric_baselines_with_pc1(df92, y),
        "df98": psychometric_baselines_with_pc1(df98, y),
    }
    print("Full-feature df92:", res["full_feature_baselines"]["df92"])
    print("Psychometric df92:", res["psychometric_baselines"]["df92"])

    # 4. Covariate-adjusted validity on df92 (Major 2.4b)
    print("Computing covariate-adjusted validity...")
    res["covariate_adjusted_validity_df92"] = {
        "chow_liu": covariate_adjusted_validity(s_cl92, y, df98),
        "ae_p85": covariate_adjusted_validity(median_scores["df92_nonlinear_p85"], y, df98),
    }
    print("Covariate-adjusted CL92 joint:", res["covariate_adjusted_validity_df92"]["chow_liu"]["joint_all_6"])
    print("Covariate-adjusted AE92 joint:", res["covariate_adjusted_validity_df92"]["ae_p85"]["joint_all_6"])

    # 5. Fairness disparities D_M and D_tilde_M on df92 (Minor 3.4)
    res["fairness_df92"] = {
        "chow_liu": fairness_metrics(s_cl92, y, df98),
        "ae_p85": fairness_metrics(median_scores["df92_nonlinear_p85"], y, df98),
    }

    # 6. Out-of-outcome ("drug-disjoint" & "substance-disjoint") downstream cleaning (Major 2.3)
    print("Computing out-of-outcome downstream drug disparities...")
    res["downstream_drug_disparity"] = drug_disparity_eval(
        raw_df, df98, y,
        {
            "cl98_insample": s_cl98,
            "ae98_insample": median_scores["df98_nonlinear_p85"],
            "cl92_comp_disjoint": s_cl92,
            "ae92_comp_disjoint": median_scores["df92_nonlinear_p85"],
            "cl79_drug_disjoint": s_cl79,
            "ae79_drug_disjoint": median_scores["df79_nodrugs_nonlinear_p85"],
            "cl65_substance_disjoint": s_cl65,
            "ae65_substance_disjoint": median_scores["df65_nosubstance_nonlinear_p85"],
        },
    )
    print("Downstream (qn48..58):", json.dumps(res["downstream_drug_disparity"]["qn48_to_qn58_11items"], indent=2))

    res["elapsed_sec"] = round(time.time() - t0, 2)
    with open(OUT_JSON, "w") as f:
        json.dump(res, f, indent=2)
    print(f"All YRBS analyses saved to {OUT_JSON} in {res['elapsed_sec']}s")


if __name__ == "__main__":
    main()
