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
        if self.arch != "linear":
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

        @tf.function(reduce_retracing=True)
        def run_epoch(x_shuf, bsz):
            n = tf.shape(x_shuf)[0]
            for start in tf.range(0, n, bsz):
                end = tf.minimum(start + bsz, n)
                train_step(x_shuf[start:end])

        self._sample_losses = sample_losses
        self._train_step = train_step
        self._run_epoch = run_epoch

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

    def _fit_and_score_linear_blas(self, X_oh, seed, epochs=5, batch_size=256, train_mask=None):
        """Exact zero-layer linear per-variable softmax AE via 3D-packed NumPy BLAS (identical init & Adam)."""
        rng_init = np.random.default_rng(seed)
        fin = self.D
        m_dim = self.bottleneck
        lim1 = np.sqrt(6.0 / (fin + m_dim))
        W1 = rng_init.uniform(-lim1, lim1, size=(fin, m_dim)).astype(np.float32)
        b1 = np.zeros((m_dim,), dtype=np.float32)

        Kmax = max(self.cards)
        W2 = np.zeros((m_dim, self.V, Kmax), dtype=np.float32)
        valid = np.zeros((1, self.V, Kmax), dtype=bool)
        for v, kv in enumerate(self.cards):
            lim2 = np.sqrt(6.0 / (m_dim + kv))
            W2[:, v, :kv] = rng_init.uniform(-lim2, lim2, size=(m_dim, kv)).astype(np.float32)
            valid[0, v, :kv] = True
        b2 = np.zeros((self.V, Kmax), dtype=np.float32)

        def _pack3(X2):
            out = np.zeros((X2.shape[0], self.V, Kmax), dtype=np.float32)
            off = 0
            for v, kv in enumerate(self.cards):
                out[:, v, :kv] = X2[:, off:off + kv]
                off += kv
            return out

        X_train = X_oh[train_mask] if train_mask is not None else X_oh
        Y_train3 = _pack3(X_train)
        n_tr = X_train.shape[0]
        inv_lk = (1.0 / self.log_cards)[None, :]

        mW1, vW1 = np.zeros_like(W1), np.zeros_like(W1)
        mb1, vb1 = np.zeros_like(b1), np.zeros_like(b1)
        W2_flat = W2.reshape(m_dim, self.V * Kmax)
        b2_flat = b2.reshape(1, self.V * Kmax)
        mW2, vW2 = np.zeros_like(W2_flat), np.zeros_like(W2_flat)
        mb2, vb2 = np.zeros_like(b2_flat), np.zeros_like(b2_flat)
        beta1, beta2, eps = 0.9, 0.999, 1e-7
        t_step = 0

        rng = np.random.default_rng(seed + 5000)
        for _ in range(epochs):
            perm = rng.permutation(n_tr)
            for start in range(0, n_tr, batch_size):
                bi = perm[start:start + batch_size]
                Xb = X_train[bi]
                Yb = Y_train3[bi]
                bsz = Xb.shape[0]

                Z = Xb @ W1 + b1
                logits = (Z @ W2_flat + b2_flat).reshape(bsz, self.V, Kmax)
                logits = np.where(valid, logits, -1e9)
                lmax = logits.max(axis=-1, keepdims=True)
                ex = np.where(valid, np.exp(logits - lmax), 0.0)
                sum_ex = ex.sum(axis=-1, keepdims=True)
                probs = ex / (sum_ex + 1e-12)
                log_probs = (logits - lmax) - np.log(sum_ex + 1e-12)
                sl = (-np.sum(Yb * log_probs, axis=-1) * inv_lk).mean(axis=1)

                if self.p >= 100.0:
                    ki = np.arange(bsz)
                else:
                    k_keep = max(1, int(np.floor(bsz * (self.p / 100.0))))
                    ki = np.argsort(sl, kind="stable")[:k_keep]
                denom = float(len(ki) * self.V)
                dlogits = ((probs[ki] - Yb[ki]) * inv_lk[:, :, None]) / denom
                dlogits_flat = dlogits.reshape(len(ki), self.V * Kmax)
                dW2 = Z[ki].T @ dlogits_flat
                db2 = dlogits_flat.sum(axis=0, keepdims=True)
                dZ = dlogits_flat @ W2_flat.T
                dW1 = Xb[ki].T @ dZ
                db1 = dZ.sum(axis=0)

                t_step += 1
                lr_t = self.lr * np.sqrt(1.0 - beta2 ** t_step) / (1.0 - beta1 ** t_step)
                mW1 = beta1 * mW1 + (1.0 - beta1) * dW1
                vW1 = beta2 * vW1 + (1.0 - beta2) * (dW1 * dW1)
                W1 -= lr_t * mW1 / (np.sqrt(vW1) + eps)

                mb1 = beta1 * mb1 + (1.0 - beta1) * db1
                vb1 = beta2 * vb1 + (1.0 - beta2) * (db1 * db1)
                b1 -= lr_t * mb1 / (np.sqrt(vb1) + eps)

                mW2 = beta1 * mW2 + (1.0 - beta1) * dW2
                vW2 = beta2 * vW2 + (1.0 - beta2) * (dW2 * dW2)
                W2_flat -= lr_t * mW2 / (np.sqrt(vW2) + eps)

                mb2 = beta1 * mb2 + (1.0 - beta1) * db2
                vb2 = beta2 * vb2 + (1.0 - beta2) * (db2 * db2)
                b2_flat -= lr_t * mb2 / (np.sqrt(vb2) + eps)

        scores = np.zeros(X_oh.shape[0], dtype=np.float32)
        for start in range(0, X_oh.shape[0], 2048):
            end = min(X_oh.shape[0], start + 2048)
            Xc = X_oh[start:end]
            Yc = _pack3(Xc)
            Zc = Xc @ W1 + b1
            lc = (Zc @ W2_flat + b2_flat).reshape(end - start, self.V, Kmax)
            lc = np.where(valid, lc, -1e9)
            lm = lc.max(axis=-1, keepdims=True)
            se = np.where(valid, np.exp(lc - lm), 0.0).sum(axis=-1, keepdims=True)
            lp = (lc - lm) - np.log(se + 1e-12)
            scores[start:end] = (-np.sum(Yc * lp, axis=-1) * inv_lk).mean(axis=1)
        return scores

    def fit_and_score(self, X_oh, seed, epochs=5, batch_size=256, train_mask=None):
        if self.arch == "linear":
            return self._fit_and_score_linear_blas(X_oh, seed, epochs=epochs, batch_size=batch_size, train_mask=train_mask)
        self.reset_weights(seed)
        rng = np.random.default_rng(seed + 5000)
        X_train = X_oh[train_mask] if train_mask is not None else X_oh
        n_tr = X_train.shape[0]
        bsz_tf = tf.constant(batch_size, dtype=tf.int32)
        for _ in range(epochs):
            perm = rng.permutation(n_tr)
            self._run_epoch(tf.constant(X_train[perm]), bsz_tf)
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


_LIKERT_WORDS = {
    "strongly disagree": 1, "disagree": 2, "neither agree nor disagree": 3,
    "neutral": 3, "neither": 3, "agree": 4, "strongly agree": 5,
    "strongly oppose": 1, "oppose": 2, "neither support nor oppose": 3,
    "support": 4, "strongly support": 5,
    "never": 1, "rarely": 2, "sometimes": 3, "often": 4, "always": 5,
}


def _to_ordinal(s: pd.Series):
    """Canonical ordinal encoder matching evaluate.detection._to_ordinal (80% non-missing threshold)."""
    nn = int(s.notna().sum())
    if nn == 0:
        return None
    num = pd.to_numeric(s, errors="coerce")
    if int(num.notna().sum()) >= 0.8 * nn:
        return num
    ext = pd.to_numeric(s.astype(str).str.extract(r"^\s*(\d+)")[0], errors="coerce")
    if int(ext.notna().sum()) >= 0.8 * nn:
        return ext
    mapped = s.astype(str).str.strip().str.lower().map(_LIKERT_WORDS)
    if int(mapped.notna().sum()) >= 0.8 * nn:
        return mapped
    return None


def _extract_ordinal_battery(df_sub: pd.DataFrame, max_levels: int = 9) -> pd.DataFrame:
    """Canonical YRBS ordinal battery matching experiment_b.fairness._battery('sadc_2017')."""
    cols = {}
    for c in df_sub.columns:
        enc = _to_ordinal(df_sub[c])
        if enc is not None and 2 <= int(pd.Series(enc).dropna().nunique()) <= max_levels:
            cols[c] = np.asarray(enc, dtype=float)
    return pd.DataFrame(cols)


def _bl_longstring(B: pd.DataFrame) -> np.ndarray:
    X = B.to_numpy(dtype=float)
    n = X.shape[0]
    out = np.full(n, np.nan)
    for i in range(n):
        best = cur = 0
        prev = np.nan
        for v in X[i]:
            if np.isnan(v):
                cur = 0
                prev = np.nan
                continue
            if v == prev:
                cur += 1
            else:
                cur = 1
                prev = v
            if cur > best:
                best = cur
        if best > 0:
            out[i] = best
    return out


def _bl_irv(B: pd.DataFrame) -> np.ndarray:
    sd = B.std(axis=1, ddof=0, skipna=True).to_numpy(dtype=float)
    sd[B.notna().sum(axis=1).to_numpy() < 2] = np.nan
    return -sd


def _bl_person_total_r(B: pd.DataFrame) -> np.ndarray:
    X = B.to_numpy(dtype=float)
    item_mean = np.nanmean(X, axis=0)
    n = X.shape[0]
    out = np.full(n, np.nan)
    for i in range(n):
        m = ~np.isnan(X[i])
        if m.sum() < 3:
            continue
        a, b = X[i][m], item_mean[m]
        if np.std(a) < 1e-9 or np.std(b) < 1e-9:
            out[i] = -1.0
            continue
        out[i] = np.corrcoef(a, b)[0, 1]
    return -out


def _bl_mahalanobis(B: pd.DataFrame) -> np.ndarray:
    X = B.to_numpy(dtype=float)
    if X.shape[1] < 2 or X.shape[0] < 3:
        return np.full(X.shape[0], np.nan)
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    Ximp = X.copy()
    Ximp[inds] = np.take(col_mean, inds[1])
    sd = Ximp.std(axis=0, ddof=0)
    keep = sd > 1e-9
    if int(keep.sum()) < 2:
        return np.full(X.shape[0], np.nan)
    Z = (Ximp[:, keep] - Ximp[:, keep].mean(axis=0)) / sd[keep]
    cov = np.cov(Z, rowvar=False)
    inv = np.linalg.pinv(cov)
    d2 = np.einsum("ij,jk,ik->i", Z, inv, Z)
    return np.sqrt(np.clip(d2, 0, None))


def _bl_even_odd(B: pd.DataFrame) -> np.ndarray:
    X = B.to_numpy(dtype=float)
    even, odd = X[:, 0::2], X[:, 1::2]
    w = min(even.shape[1], odd.shape[1])
    even, odd = even[:, :w], odd[:, :w]
    n = X.shape[0]
    out = np.full(n, np.nan)
    for i in range(n):
        m = ~np.isnan(even[i]) & ~np.isnan(odd[i])
        if m.sum() < 3:
            continue
        a, b = even[i][m], odd[i][m]
        if np.std(a) < 1e-9 or np.std(b) < 1e-9:
            out[i] = -1.0
            continue
        out[i] = np.corrcoef(a, b)[0, 1]
    return -out


def psychometric_baselines_with_pc1(df_sub, y):
    B = _extract_ordinal_battery(df_sub).astype(float)
    ord_cols = list(B.columns)
    J = len(ord_cols)

    ls = _bl_longstring(B)
    irv = _bl_irv(B)
    pt_raw = _bl_person_total_r(B)
    md = _bl_mahalanobis(B)
    eo_raw = _bl_even_odd(B)

    # Codebook-free PC1 sign alignment matching run_all_snap_experiments.py
    B_imp = B.apply(lambda c: c.fillna(c.mean()), axis=0).to_numpy(dtype=float)
    sd = np.where(B_imp.std(axis=0) > 1e-8, B_imp.std(axis=0), 1.0)
    Z = (B_imp - B_imp.mean(axis=0)) / sd
    gz = (Z.T @ Z) / max(len(Z) - 1, 1)
    _, evz = np.linalg.eigh(0.5 * (gz + gz.T))
    w = evz[:, -1].copy()
    if (w >= 0).sum() < len(w) / 2.0:
        w = -w

    B_flip = B.copy()
    neg_items = []
    for j, col in enumerate(ord_cols):
        if w[j] < 0:
            cmin, cmax = np.nanmin(B[col]), np.nanmax(B[col])
            B_flip[col] = cmax + cmin - B[col]
            neg_items.append(col)

    pt_pc1 = _bl_person_total_r(B_flip)
    eo_pc1 = _bl_even_odd(B_flip)

    def _auc_valid(arr):
        m = np.isfinite(arr)
        if m.sum() < 10 or not (0 < y[m].sum() < m.sum()):
            return None
        return float(roc_auc_score(y[m], arr[m]))

    return {
        "n_ordinal_items": J,
        "ordinal_items": ord_cols,
        "n_pc1_flipped": len(neg_items),
        "pc1_flipped_items": neg_items,
        "ls_auc": _auc_valid(ls),
        "irv_auc": _auc_valid(irv),
        "md_ord_auc": _auc_valid(md),
        "pt_raw_auc": _auc_valid(pt_raw),
        "pt_pc1_auc": _auc_valid(pt_pc1),
        "eo_raw_auc": _auc_valid(eo_raw),
        "eo_pc1_auc": _auc_valid(eo_pc1),
        "lz_auc": None,
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

    def _save(obj):
        os.makedirs(os.path.dirname(os.path.abspath(OUT_JSON)), exist_ok=True)
        with open(OUT_JSON, "w") as f:
            json.dump(obj, f, indent=2)

    resume = "--resume" in sys.argv
    if resume and os.path.exists(OUT_JSON):
        res = json.load(open(OUT_JSON))
    else:
        res = {}

    seeds = [1, 2, 3, 4, 5]

    if not (resume and "downstream_drug_disparity" in res):
        res["provenance"] = {
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
        _save(res)

    if "--skip-converged-linear" not in sys.argv:
        from concurrent.futures import ThreadPoolExecutor

        # 7. Converged 50-epoch lr=1e-3 linear refits (Table 2 note)
        t_step7 = time.time()
        print("Running 50-epoch lr=1e-3 converged linear refits...")
        lin_conv = dict(res.get("linear_50ep_lr1e3", {})) if resume else {}
        for dtag, X_mat, cards_vec in [("df98", X98, cards98), ("df92", X92, cards92)]:
            lin_conv.setdefault(dtag, {})
            for bneck in [2, 8]:
                for pval in [100, 85]:
                    key = f"linear_m{bneck}_p{pval}_50ep_lr1e3"
                    if resume and key in lin_conv[dtag]:
                        continue
                    tr_lin = BlockSoftmaxAE(cards_vec, arch="linear", bottleneck=bneck, lr=1e-3, p=float(pval))

                    def _one_seed(sd, _tr=tr_lin, _X=X_mat):
                        rng_sp = np.random.default_rng(100 + sd)
                        tr_mask = rng_sp.random(_X.shape[0]) >= 0.2
                        sc = _tr.fit_and_score(_X, seed=1000 + sd, epochs=50, batch_size=256, train_mask=tr_mask)
                        return round(float(roc_auc_score(y, sc)), 4)

                    with ThreadPoolExecutor(max_workers=2) as pool:
                        s_aucs = list(pool.map(_one_seed, seeds))
                    lin_conv[dtag][key] = {
                        "median": round(float(np.median(s_aucs)), 4),
                        "mean": round(float(np.mean(s_aucs)), 4),
                        "seeds": s_aucs,
                    }
                    print(f"  {dtag} {key}: median={lin_conv[dtag][key]['median']:.4f} seeds={s_aucs}")
                    res["linear_50ep_lr1e3"] = lin_conv
                    _save(res)
        res["elapsed_sec"] = round(
            (res.get("elapsed_sec", 0.0) + (time.time() - t_step7)) if resume else (time.time() - t0), 2
        )
        _save(res)

    print(f"All YRBS analyses saved to {OUT_JSON} in {res['elapsed_sec']}s")


if __name__ == "__main__":
    main()
