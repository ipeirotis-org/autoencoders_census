"""
Unified, fast, exact execution script for all multi-dataset [SNAP] revision experiments
on a hydrated 2026-10-23 paper-data snapshot (SNAP_REPO).

Key fidelity guarantees:
- Reconciles `public_opinion` (`mastroianni2022`) to the manuscript's exact N = 1,003
  completed consenting surveys (`h = 29` slider-check failers, `974` passers; commit 99c3879),
  including its `language` field for Major 2.4b.
- Uses the exact 92-variable composite-disjoint `sadc_2017` (`kann2018`) dataset
  (dropping the 6 YRBS composite source columns).
- Records BOTH native orientation AUC (Table 2 convention) and label-oriented AUC
  (`max(auc, 1 - auc)`) for every detector and baseline.
- Uses exact vectorized NumPy BLAS Adam + EarlyStopping for 0-layer Linear AE (`Lin_100`, `Lin_85`)
  and graph-cached `BlockSoftmaxHead` Keras models for 1-layer Non-Linear AE (`AE_100`, `AE_85`).

Writes incremental JSON checkpoints to scratch/snap_revision_results.json.
"""
from __future__ import annotations
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORK_REPO = os.environ.get("SNAP_REPO", os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..")))
sys.path.insert(0, WORK_REPO)
os.chdir(WORK_REPO)

import numpy as np
import pandas as pd
import yaml
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense, BatchNormalization, Dropout
from keras.regularizers import l2
from scipy import stats
from sklearn.metrics import roc_auc_score, mutual_info_score

from utils import set_seed, define_necessary_elements, training_batch_size
from dataset.loader import DataLoader
from main import (
    prepare_for_training,
    prepare_for_categorical,
    prepare_for_model,
    _clean_for_saved_vectorizer,
    _compute_attr_layout,
)
from evaluate.detection import (
    aligned_labels,
    aligned_battery,
    CHECKS,
    _relevant,
    _read_raw,
)
from evaluate import baselines as bl
from chow_liu_rank import CLTree
from experiment_b import fairness as F
from revision_experiments import lib_r2 as L

OUT_JSON = os.environ.get("OUT_JSON", os.path.join(SCRIPT_DIR, "results", "snap_revision_results.json"))
CHECKPOINT_JSON = os.environ.get("CHECKPOINT_JSON", "")
YRBS_JSON = os.environ.get("YRBS_JSON", os.path.join(SCRIPT_DIR, "results", "yrbs_revision_results.json"))

DS_8 = [
    "attention_check",
    "bot_bot_mturk",
    "inattentive",
    "moral_data",
    "mturk_ethics",
    "pennycook_1",
    "public_opinion",
    "racial_data",
]
CITE = {
    "attention_check": "uhalt2020",
    "bot_bot_mturk": "buchanan2018",
    "inattentive": "alvarez2019",
    "moral_data": "ogrady2019",
    "mturk_ethics": "moss2023",
    "pennycook_1": "pennycook2020",
    "public_opinion": "mastroianni2022",
    "racial_data": "ivanov2021",
    "sadc_2017": "kann2018",
}

SADC_COMP_6 = [
    "carrot_eating",
    "green _salad_eating",
    "fruit_juice_drinking",
    "fruit_eating",
    "stheight_cat",
    "stweight_cat",
]


def save_ckpt(results: dict):
    target = CHECKPOINT_JSON or OUT_JSON
    os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
    with open(target, "w") as f:
        json.dump(results, f, indent=2)


def auc_pair(y: np.ndarray, s: np.ndarray):
    nat = L.auc_native(y, s)
    if not np.isfinite(nat):
        return None, None
    return round(float(nat), 4), round(float(max(nat, 1.0 - nat)), 4)


def public_opinion_completer_mask():
    raw = _read_raw("public_opinion")
    keep = (
        raw["Finished"].astype(str).str.strip().isin(["1", "1.0", "True", "true"])
        & (raw["consent"] == "I consent to take part in this study")
    ).to_numpy()
    sub_raw = raw.loc[keep].reset_index(drop=True)
    a1 = pd.to_numeric(sub_raw["attention_1"], errors="coerce") == 1
    a2 = pd.to_numeric(sub_raw["attention_2"], errors="coerce") == 1
    y = (~(a1 & a2)).astype(int).to_numpy()
    return keep, sub_raw, y


def load_dataset_bundle(ds: str):
    """Return `(data_df, y_primary, battery_df_or_None)` reconciled to manuscript specs."""
    if ds == "sadc_2017":
        dc, rc, ic, adc, arc, aic = define_necessary_elements("sadc_2017", None, None, None)
        dl = DataLoader(dc, rc, ic, additional_drop_columns=adc, additional_rename_columns=arc, additional_columns_of_interest=aic)
        y = dl.find_outlier_data_sadc_2017("sadc_2017", ["outlier"])["outlier"].values.astype(int)
        data, _ = dl.load_data("sadc_2017")
        cat = prepare_for_categorical(data.copy()).reset_index(drop=True)
        drop6 = [c for c in SADC_COMP_6 if c in cat.columns]
        cat92 = cat.drop(columns=drop6)
        bat = F._battery("sadc_2017")
        bat = bat.drop(columns=[c for c in drop6 if c in bat.columns], errors="ignore")
        return cat92, y, bat

    dc, rc, ic, adc, arc, aic = define_necessary_elements(ds, None, None, None)
    dl = DataLoader(dc, rc, ic, additional_drop_columns=adc, additional_rename_columns=arc, additional_columns_of_interest=aic)
    data, _ = dl.load_data(ds)
    bat = aligned_battery(ds)
    if ds == "public_opinion":
        keep, _, y = public_opinion_completer_mask()
        data = data.loc[keep].reset_index(drop=True)
        bat = bat.loc[keep].reset_index(drop=True)
        return data, y, bat

    y = L.primary_label(ds)
    return data.reset_index(drop=True), y, bat.reset_index(drop=True)


def pack_3d(X2d: np.ndarray, card: list[int]) -> np.ndarray:
    N = X2d.shape[0]
    V = len(card)
    Kmax = max(card)
    out = np.zeros((N, V, Kmax), dtype=np.float32)
    s = 0
    for v, k in enumerate(card):
        out[:, v, :k] = X2d[:, s : s + k]
        s += k
    return out


def numpy_linear_ae_seeds(
    data_df: pd.DataFrame,
    m_dim: int,
    percentile: int,
    batch_size: int,
    epochs: int = 120,
    lr: float = 1e-3,
    seeds=(1, 2, 3, 4, 5),
) -> list[np.ndarray]:
    """Exact 0-layer Linear AE (`latent_activation='linear'`) trained with Adam + Percentile Loss + EarlyStopping."""
    vt = {c: "categorical" for c in data_df.columns}
    errs = []
    beta1, beta2, eps = 0.9, 0.999, 1e-7

    for seed in seeds:
        set_seed(seed)
        rng = np.random.RandomState(seed)
        _, Xtr_df, Xte_df, vec, card = prepare_for_training(data_df, vt, test_size=0.2)
        Xtr = Xtr_df.to_numpy(dtype=np.float32)
        Xte = Xte_df.to_numpy(dtype=np.float32)
        Ytr3 = pack_3d(Xtr, card)
        Yte3 = pack_3d(Xte, card)

        D = Xtr.shape[1]
        V = len(card)
        Kmax = max(card)
        valid = np.zeros((1, V, Kmax), dtype=bool)
        for v, k in enumerate(card):
            valid[0, v, :k] = True
        inv_lk = np.array([1.0 / np.log(max(int(k), 2)) for k in card], dtype=np.float32)[None, :]

        lim1 = np.sqrt(6.0 / (D + m_dim))
        W1 = rng.uniform(-lim1, lim1, size=(D, m_dim)).astype(np.float32)
        b1 = np.zeros((m_dim,), dtype=np.float32)
        W2 = np.zeros((m_dim, V, Kmax), dtype=np.float32)
        for v, k in enumerate(card):
            lim2 = np.sqrt(6.0 / (m_dim + k))
            W2[:, v, :k] = rng.uniform(-lim2, lim2, size=(m_dim, k)).astype(np.float32)
        b2 = np.zeros((V, Kmax), dtype=np.float32)

        mW1, vW1 = np.zeros_like(W1), np.zeros_like(W1)
        mb1, vb1 = np.zeros_like(b1), np.zeros_like(b1)
        mW2, vW2 = np.zeros_like(W2), np.zeros_like(W2)
        mb2, vb2 = np.zeros_like(b2), np.zeros_like(b2)
        t_step = 0

        def _forward_probs(X2):
            Z = X2 @ W1 + b1
            logits = np.einsum("bm,mvk->bvk", Z, W2, optimize=True) + b2[None, :, :]
            logits = np.where(valid, logits, -1e9)
            lmax = logits.max(axis=-1, keepdims=True)
            ex = np.where(valid, np.exp(logits - lmax), 0.0)
            P = ex / ex.sum(axis=-1, keepdims=True)
            return Z, P

        def _loss_val(Y3, P):
            Pc = np.clip(P, 1e-7, 1.0 - 1e-7)
            rl = (-np.sum(Y3 * np.log(Pc), axis=-1) * inv_lk).mean(axis=1)
            n = len(rl)
            k = n if percentile == 100 else max(int((percentile / 100.0) * n), 1)
            if k >= n:
                return float(rl.mean())
            idx = np.argpartition(rl, k - 1)[:k]
            return float(rl[idx].mean())

        best_val = np.inf
        wait = 0
        best_weights = (W1.copy(), b1.copy(), W2.copy(), b2.copy())
        ntr = len(Xtr)

        for _ in range(epochs):
            perm = rng.permutation(ntr)
            for start in range(0, ntr, batch_size):
                bi = perm[start : start + batch_size]
                Xb, Yb = Xtr[bi], Ytr3[bi]
                B = len(bi)
                Z, P = _forward_probs(Xb)
                Pc = np.clip(P, 1e-7, 1.0 - 1e-7)
                rl = (-np.sum(Yb * np.log(Pc), axis=-1) * inv_lk).mean(axis=1)
                k_keep = B if percentile == 100 else max(int((percentile / 100.0) * B), 1)
                if k_keep >= B:
                    ki = np.arange(B)
                else:
                    ki = np.argpartition(rl, k_keep - 1)[:k_keep]

                # Exact gradient on kept samples
                G = (P[ki] - Yb[ki]) * (inv_lk[:, :, None] / (len(ki) * V))
                G = np.where(valid, G, 0.0).astype(np.float32)
                Zs, Xs = Z[ki], Xb[ki]
                gW2 = np.einsum("bm,bvk->mvk", Zs, G, optimize=True)
                gb2 = G.sum(axis=0)
                dZ = np.einsum("bvk,mvk->bm", G, W2, optimize=True)
                gW1 = Xs.T @ dZ
                gb1 = dZ.sum(axis=0)

                t_step += 1
                bc1 = 1.0 - beta1**t_step
                bc2 = 1.0 - beta2**t_step
                lr_t = lr * np.sqrt(bc2) / bc1

                mW1 = beta1 * mW1 + (1.0 - beta1) * gW1
                vW1 = beta2 * vW1 + (1.0 - beta2) * (gW1 * gW1)
                W1 -= lr_t * mW1 / (np.sqrt(vW1) + eps)

                mb1 = beta1 * mb1 + (1.0 - beta1) * gb1
                vb1 = beta2 * vb1 + (1.0 - beta2) * (gb1 * gb1)
                b1 -= lr_t * mb1 / (np.sqrt(vb1) + eps)

                mW2 = beta1 * mW2 + (1.0 - beta1) * gW2
                vW2 = beta2 * vW2 + (1.0 - beta2) * (gW2 * gW2)
                W2 -= lr_t * mW2 / (np.sqrt(vW2) + eps)

                mb2 = beta1 * mb2 + (1.0 - beta1) * gb2
                vb2 = beta2 * vb2 + (1.0 - beta2) * (gb2 * gb2)
                b2 -= lr_t * mb2 / (np.sqrt(vb2) + eps)

            # Val loss in batches (matches Keras validation_data batching)
            val_losses = []
            val_weights = []
            for s0 in range(0, len(Xte), batch_size):
                Xvb, Yvb = Xte[s0 : s0 + batch_size], Yte3[s0 : s0 + batch_size]
                _, Pv = _forward_probs(Xvb)
                val_losses.append(_loss_val(Yvb, Pv))
                val_weights.append(len(Xvb))
            vloss = float(np.average(val_losses, weights=val_weights))
            if vloss < best_val - 1e-7:
                best_val = vloss
                best_weights = (W1.copy(), b1.copy(), W2.copy(), b2.copy())
                wait = 0
            else:
                wait += 1
                if wait >= 10:
                    break

        W1, b1, W2, b2 = best_weights
        proj = _clean_for_saved_vectorizer(data_df.copy(), vec)
        Xfull = vec.transform(proj).to_numpy(dtype=np.float32)
        ac, _, _ = _compute_attr_layout(vec, proj.columns)
        Xf3 = pack_3d(Xfull, ac)
        _, Pf = _forward_probs(Xfull)
        Pc = np.clip(Pf, 1e-7, 1.0 - 1e-7)
        ce = -np.sum(Xf3 * np.log(Pc), axis=-1) * inv_lk
        obs = Xf3.sum(axis=-1) > 0.5
        ce = np.where(obs, ce, 1.0)
        errs.append(ce.mean(axis=1))
    return errs


class BlockSoftmaxHead(keras.layers.Layer):
    def __init__(self, card: list[int], **kwargs):
        super().__init__(**kwargs)
        self.card = list(card)
        self.V = len(card)
        self.Kmax = max(card)
        mask = np.zeros((1, self.V, self.Kmax), dtype=bool)
        for v, k in enumerate(card):
            mask[0, v, :k] = True
        self.valid_mask = tf.constant(mask)

    def build(self, input_shape):
        in_dim = int(input_shape[-1])
        self.in_dim = in_dim
        self.W = self.add_weight(shape=(in_dim, self.V, self.Kmax), initializer="zeros", trainable=True, name="W")
        self.b = self.add_weight(shape=(self.V, self.Kmax), initializer="zeros", trainable=True, name="b")
        self.reinit(np.random.RandomState(0))

    def reinit(self, rng: np.random.RandomState):
        w_init = np.zeros((self.in_dim, self.V, self.Kmax), dtype=np.float32)
        for v, k in enumerate(self.card):
            lim = np.sqrt(6.0 / (self.in_dim + k))
            w_init[:, v, :k] = rng.uniform(-lim, lim, size=(self.in_dim, k)).astype(np.float32)
        self.W.assign(w_init)
        self.b.assign(np.zeros((self.V, self.Kmax), dtype=np.float32))

    def call(self, h):
        logits = tf.einsum("bi,ivk->bvk", h, self.W) + self.b[None, :, :]
        logits = tf.where(self.valid_mask, logits, tf.constant(-1e9, dtype=tf.float32))
        return tf.nn.softmax(logits, axis=-1)


class VecPercentileLoss(keras.losses.Loss):
    def __init__(self, card: list[int], percentile: int = 85):
        super().__init__(reduction=keras.losses.Reduction.NONE)
        self.percentile = int(percentile)
        self.inv_log_k = tf.constant([1.0 / np.log(max(int(k), 2)) for k in card], dtype=tf.float32)[None, :]

    def call(self, y_true, y_pred):
        p = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -tf.reduce_sum(y_true * tf.math.log(p), axis=-1) * self.inv_log_k
        row_loss = tf.reduce_mean(ce, axis=1)
        n = tf.shape(row_loss)[0]
        k = (
            n
            if self.percentile == 100
            else tf.maximum(tf.cast((self.percentile / 100.0) * tf.cast(n, tf.float32), tf.int32), 1)
        )
        smallest, _ = tf.nn.top_k(-row_loss, k=k)
        return tf.reduce_mean(-smallest)


_KERAS_CACHE: dict = {}


def _reinit_keras_model(model: keras.Model, seed: int):
    rng = np.random.RandomState(seed)
    for layer in model.layers:
        if isinstance(layer, BlockSoftmaxHead):
            layer.reinit(rng)
        elif isinstance(layer, Dense):
            k_sh = layer.kernel.shape
            lim = np.sqrt(6.0 / (int(k_sh[0]) + int(k_sh[1])))
            layer.kernel.assign(rng.uniform(-lim, lim, size=k_sh).astype(np.float32))
            if layer.bias is not None:
                layer.bias.assign(np.zeros(layer.bias.shape, dtype=np.float32))
        elif isinstance(layer, BatchNormalization):
            layer.gamma.assign(np.ones(layer.gamma.shape, dtype=np.float32))
            layer.beta.assign(np.zeros(layer.beta.shape, dtype=np.float32))
            layer.moving_mean.assign(np.zeros(layer.moving_mean.shape, dtype=np.float32))
            layer.moving_variance.assign(np.ones(layer.moving_variance.shape, dtype=np.float32))
    for v in model.optimizer.variables:
        v.assign(tf.zeros_like(v))


def fast_keras_ae_seeds(data_df: pd.DataFrame, cfg: dict, batch_size: int, seeds=(1, 2, 3, 4, 5)) -> list[np.ndarray]:
    """Graph-cached 1-layer Non-Linear AE runner (compiles TF graph once per `(D, card, cfg)`)."""
    vt = {c: "categorical" for c in data_df.columns}
    errs = []
    cfg_sig = tuple(sorted((k, str(v)) for k, v in cfg.items()))
    for s in seeds:
        set_seed(s)
        _, Xtr_df, Xte_df, vec, card = prepare_for_training(data_df, vt, test_size=0.2)
        Xtr = Xtr_df.to_numpy(dtype=np.float32)
        Xte = Xte_df.to_numpy(dtype=np.float32)
        Ytr3 = pack_3d(Xtr, card)
        Yte3 = pack_3d(Xte, card)

        key = (Xtr.shape[1], tuple(card), cfg_sig)
        if key not in _KERAS_CACHE:
            inp = Input(shape=(Xtr.shape[1],))
            x = inp
            for i in range(cfg.get("encoder_layers", 0)):
                reg = l2(cfg.get(f"encoder_l2_{i+1}", 0.0)) if cfg.get(f"encoder_l2_{i+1}", 0.0) > 0 else None
                x = Dense(
                    cfg[f"encoder_units_{i+1}"],
                    activation=cfg[f"encoder_activation_{i+1}"],
                    kernel_regularizer=reg,
                )(x)
                if cfg.get(f"encoder_dropout_{i+1}", 0.0) > 0:
                    x = Dropout(cfg[f"encoder_dropout_{i+1}"])(x)
                if cfg.get(f"encoder_batch_norm_{i+1}", False):
                    x = BatchNormalization()(x)
            z = Dense(cfg["latent_space_dim"], activation=cfg.get("latent_activation", "relu"))(x)
            h = z
            for i in range(cfg.get("decoder_layers", 0)):
                reg = l2(cfg.get(f"decoder_l2_{i+1}", 0.0)) if cfg.get(f"decoder_l2_{i+1}", 0.0) > 0 else None
                h = Dense(
                    cfg[f"decoder_units_{i+1}"],
                    activation=cfg[f"decoder_activation_{i+1}"],
                    kernel_regularizer=reg,
                )(h)
                if cfg.get(f"decoder_dropout_{i+1}", 0.0) > 0:
                    h = Dropout(cfg[f"decoder_dropout_{i+1}"])(h)
                if cfg.get(f"decoder_batch_norm_{i+1}", False):
                    h = BatchNormalization()(h)
            out = BlockSoftmaxHead(card)(h)
            model = keras.Model(inp, out)
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=cfg.get("learning_rate", 1e-3)),
                loss=VecPercentileLoss(card, cfg["percentile"]),
            )
            _KERAS_CACHE[key] = model
        else:
            model = _KERAS_CACHE[key]

        _reinit_keras_model(model, s)
        es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        model.fit(
            Xtr,
            Ytr3,
            epochs=cfg.get("epochs", 120),
            batch_size=batch_size,
            verbose=0,
            validation_data=(Xte, Yte3),
            callbacks=[es],
        )

        proj = _clean_for_saved_vectorizer(data_df.copy(), vec)
        Xfull = vec.transform(proj).to_numpy(dtype=np.float32)
        ac, _, _ = _compute_attr_layout(vec, proj.columns)
        Xfull3 = pack_3d(Xfull, ac)
        preds3 = model(Xfull, training=False).numpy()
        p_clip = np.clip(preds3, 1e-7, 1.0 - 1e-7)
        inv_lk = np.array([1.0 / np.log(max(int(k), 2)) for k in ac], dtype=np.float32)[None, :]
        ce = -np.sum(Xfull3 * np.log(p_clip), axis=-1) * inv_lk
        obs = Xfull3.sum(axis=-1) > 0.5
        ce = np.where(obs, ce, 1.0)
        errs.append(ce.mean(axis=1))
    return errs


def run_block_1_ablation(results: dict):
    t0 = time.time()
    print("\n" + "=" * 78)
    print("BLOCK 1 [SNAP 2.1b & 3.5]: 2x2 {Linear, Non-Linear} x {p=100, p=85} + Hyperparams")
    print("=" * 78, flush=True)
    hp_table = {}
    ablation = {}

    for ds in DS_8:
        cite = CITE[ds]
        best = yaml.safe_load(open(f"cache/_tuned_{ds}_ae/best_hyperparameters.yaml"))
        m_dim = int(best["latent_space_dim"])
        hp_table[cite] = {
            "m": m_dim,
            "enc_units": int(best["encoder_units_1"]),
            "enc_act": str(best["encoder_activation_1"]),
            "enc_bn": bool(best.get("encoder_batch_norm_1", False)),
            "enc_drop": round(float(best.get("encoder_dropout_1", 0.0)), 2),
            "enc_l2": float(best.get("encoder_l2_1", 0.0)),
            "dec_units": int(best["decoder_units_1"]),
            "dec_act": str(best["decoder_activation_1"]),
            "dec_bn": bool(best.get("decoder_batch_norm_1", False)),
            "dec_drop": round(float(best.get("decoder_dropout_1", 0.0)), 2),
            "dec_l2": float(best.get("decoder_l2_1", 0.0)),
            "latent_act": str(best.get("latent_activation", "relu")),
            "lr": float(best.get("learning_rate", 1e-3)),
        }

        data_df, y, _ = load_dataset_bundle(ds)
        bs = training_batch_size(ds)

        # Chow-Liu
        if ds == "public_opinion":
            cat_po = prepare_for_categorical(data_df.copy()).reset_index(drop=True)
            cl_sc = (1.0 - CLTree(alpha=1.0).fit(cat_po, random_state=2).score_dataframe(cat_po)["pct"]).to_numpy()
        else:
            cl_sc = (1.0 - pd.read_csv(f"cache/_fix9_{ds}_cl/errors.csv")["pct"]).to_numpy()
        cl_nat, cl_ori = auc_pair(y, cl_sc)

        # Linear AE 5 seeds at p=100 and p=85 (fast exact NumPy BLAS)
        lin100_errs = numpy_linear_ae_seeds(data_df, m_dim=m_dim, percentile=100, batch_size=bs)
        lin85_errs = numpy_linear_ae_seeds(data_df, m_dim=m_dim, percentile=85, batch_size=bs)
        lin100_nats = [auc_pair(y, e)[0] for e in lin100_errs]
        lin100_oris = [auc_pair(y, e)[1] for e in lin100_errs]
        lin85_nats = [auc_pair(y, e)[0] for e in lin85_errs]
        lin85_oris = [auc_pair(y, e)[1] for e in lin85_errs]

        # Cached Lin_100 (or reconciled Table 2 0.62 on mastroianni2022)
        if ds == "public_opinion":
            keep, _, _ = public_opinion_completer_mask()
            lin100_c_nat, _lin100_c_ori = auc_pair(y, pd.read_csv(f"cache/_lin0_{ds}/errors.csv")["error"].to_numpy()[keep])
            cfg100 = dict(best); cfg100.update(epochs=120, percentile=100)
            cfg85 = dict(best); cfg85.update(epochs=120, percentile=85)
            ae100_errs = fast_keras_ae_seeds(data_df, cfg100, batch_size=bs)
            ae85_errs = fast_keras_ae_seeds(data_df, cfg85, batch_size=bs)
        else:
            lin100_c_nat, _lin100_c_ori = auc_pair(y, pd.read_csv(f"cache/_lin0_{ds}/errors.csv")["error"].to_numpy())
            ae100_errs = L.load_multiseed(ds, p=100) or [pd.read_csv(f"cache/_tuned_{ds}_ae_p100/errors.csv")["error"].to_numpy()]
            ae85_errs = L.load_multiseed(ds, p=85)

        ae100_nat = float(np.median([auc_pair(y, e)[0] for e in ae100_errs]))
        ae100_ori = float(np.median([auc_pair(y, e)[1] for e in ae100_errs]))
        ae85_nat = float(np.median([auc_pair(y, e)[0] for e in ae85_errs]))
        ae85_ori = float(np.median([auc_pair(y, e)[1] for e in ae85_errs]))

        ablation[cite] = {
            "N": int(len(data_df)),
            "m": m_dim,
            "CL_native": cl_nat,
            "CL_oriented": cl_ori,
            "Lin_100_cached_native": lin100_c_nat,
            "Lin_100_s1_native": round(float(lin100_nats[0]), 4),
            "Lin_100_med5_native": round(float(np.median(lin100_nats)), 4),
            "Lin_100_med5_oriented": round(float(np.median(lin100_oris)), 4),
            "Lin_85_s1_native": round(float(lin85_nats[0]), 4),
            "Lin_85_med5_native": round(float(np.median(lin85_nats)), 4),
            "Lin_85_med5_oriented": round(float(np.median(lin85_oris)), 4),
            "AE_100_med_native": round(ae100_nat, 4),
            "AE_100_med_oriented": round(ae100_ori, 4),
            "AE_85_med_native": round(ae85_nat, 4),
            "AE_85_med_oriented": round(ae85_ori, 4),
        }
        r = ablation[cite]
        print(
            f"  {cite:16s} (N={r['N']:4d}, m={m_dim}): CL={r['CL_native']:.3f} | "
            f"Lin100(cache/med5)={r['Lin_100_cached_native']:.3f}/{r['Lin_100_med5_native']:.3f} | "
            f"Lin85(med5)={r['Lin_85_med5_native']:.3f} | "
            f"AE100={r['AE_100_med_native']:.3f} | AE85={r['AE_85_med_native']:.3f}",
            flush=True,
        )

    # Verified 92-var composite-disjoint kann2018 (`sadc_2017`):
    # All four {Linear, Non-Linear} x {p=100, p=85} runs share identical settings
    # from config/simple_autoencoder.yaml (m=2, B=256, lr=1e-4, epochs=5, seeds 1..5).
    cat92, y_sadc, _ = load_dataset_bundle("sadc_2017")
    cl92 = (1.0 - CLTree(alpha=1.0).fit(cat92, mi_subsample=10000, random_state=2).score_dataframe(cat92)["pct"]).to_numpy()
    cl92_nat, cl92_ori = auc_pair(y_sadc, cl92)

    hp_table["kann2018"] = {
        "m": 2,
        "enc_units": 128,
        "enc_act": "relu",
        "enc_bn": False,
        "enc_drop": 0.0,
        "enc_l2": 0.0,
        "dec_units": 128,
        "dec_act": "relu",
        "dec_bn": False,
        "dec_drop": 0.0,
        "dec_l2": 0.0,
        "latent_act": "relu",
        "lr": 0.0001,
        "note": "Fixed conservative preset (5 epochs, B=256, lr=1e-4, m=2)",
    }
    if os.path.exists(YRBS_JSON) and "--recompute-yrbs-ae" not in sys.argv:
        yrbs_full = json.load(open(YRBS_JSON))
        yrbs_ae = yrbs_full["ae_ablation"]["df92"]
        lin100_yrbs_seeds = [float(x) for x in yrbs_ae["linear_m2_p100"]["seed_aucs"]]
        lin85_yrbs_seeds = [float(x) for x in yrbs_ae["linear_m2_p85"]["seed_aucs"]]
        lin100_med_nat = round(float(yrbs_ae["linear_m2_p100"]["median_auc"]), 4)
        lin85_med_nat = round(float(yrbs_ae["linear_m2_p85"]["median_auc"]), 4)
        ae100_med_nat = round(float(yrbs_ae["nonlinear_p100"]["median_auc"]), 4)
        ae85_med_nat = round(float(yrbs_ae["nonlinear_p85"]["median_auc"]), 4)
        lin100_s1_nat = round(lin100_yrbs_seeds[0], 4)
        lin85_s1_nat = round(lin85_yrbs_seeds[0], 4)
    else:
        yrbs_base_cfg = {
            "encoder_layers": 1,
            "encoder_units_1": 128,
            "encoder_activation_1": "relu",
            "latent_space_dim": 2,
            "latent_activation": "relu",
            "decoder_layers": 1,
            "decoder_units_1": 128,
            "decoder_activation_1": "relu",
            "learning_rate": 1e-4,
            "epochs": 5,
        }
        lin100_sadc = numpy_linear_ae_seeds(cat92, m_dim=2, percentile=100, batch_size=256, epochs=5, lr=1e-4, seeds=(1, 2, 3, 4, 5))
        lin85_sadc = numpy_linear_ae_seeds(cat92, m_dim=2, percentile=85, batch_size=256, epochs=5, lr=1e-4, seeds=(1, 2, 3, 4, 5))
        ae100_sadc = fast_keras_ae_seeds(cat92, {**yrbs_base_cfg, "percentile": 100}, batch_size=256, seeds=(1, 2, 3, 4, 5))
        ae85_sadc = fast_keras_ae_seeds(cat92, {**yrbs_base_cfg, "percentile": 85}, batch_size=256, seeds=(1, 2, 3, 4, 5))
        lin100_s1_nat = round(float(auc_pair(y_sadc, lin100_sadc[0])[0]), 4)
        lin85_s1_nat = round(float(auc_pair(y_sadc, lin85_sadc[0])[0]), 4)
        lin100_med_nat = round(float(np.median([auc_pair(y_sadc, e)[0] for e in lin100_sadc])), 4)
        lin85_med_nat = round(float(np.median([auc_pair(y_sadc, e)[0] for e in lin85_sadc])), 4)
        ae100_med_nat = round(float(np.median([auc_pair(y_sadc, e)[0] for e in ae100_sadc])), 4)
        ae85_med_nat = round(float(np.median([auc_pair(y_sadc, e)[0] for e in ae85_sadc])), 4)

    ablation["kann2018"] = {
        "N": int(len(cat92)),
        "m": 2,
        "CL_native": cl92_nat,
        "CL_oriented": cl92_ori,
        "Lin_100_cached_native": lin100_med_nat,
        "Lin_100_s1_native": lin100_s1_nat,
        "Lin_100_med5_native": lin100_med_nat,
        "Lin_100_med5_oriented": round(max(lin100_med_nat, 1.0 - lin100_med_nat), 4),
        "Lin_85_s1_native": lin85_s1_nat,
        "Lin_85_med5_native": lin85_med_nat,
        "Lin_85_med5_oriented": round(max(lin85_med_nat, 1.0 - lin85_med_nat), 4),
        "AE_100_med_native": ae100_med_nat,
        "AE_100_med_oriented": round(max(ae100_med_nat, 1.0 - ae100_med_nat), 4),
        "AE_85_med_native": ae85_med_nat,
        "AE_85_med_oriented": round(max(ae85_med_nat, 1.0 - ae85_med_nat), 4),
    }
    print(f"  kann2018         (N={len(cat92):5d}, m=2): {ablation['kann2018']}", flush=True)

    keys9 = list(ablation.keys())
    means = {
        col: round(float(np.mean([ablation[k][col] for k in keys9])), 4)
        for col in [
            "CL_native",
            "CL_oriented",
            "Lin_100_cached_native",
            "Lin_100_med5_native",
            "Lin_100_med5_oriented",
            "Lin_85_med5_native",
            "Lin_85_med5_oriented",
            "AE_100_med_native",
            "AE_100_med_oriented",
            "AE_85_med_native",
            "AE_85_med_oriented",
        ]
    }
    print(f"  9-SURVEY MEANS ({time.time()-t0:.1f}s):", means, flush=True)
    results["hyperparameters"] = hp_table
    results["ablation_2x2"] = {"per_dataset": ablation, "means_9": means}
    save_ckpt(results)


def onehot_classical_scores(X: np.ndarray, m_dim: int):
    X = np.asarray(X, dtype=np.float64)
    N, D = X.shape
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    cov = (Xc.T @ Xc) / max(N - 1, 1)
    cov = 0.5 * (cov + cov.T)

    evals, evecs = np.linalg.eigh(cov)
    tol = 1e-6 * max(float(evals[-1]), 1e-12)
    pos = evals > tol
    proj_pos = Xc @ evecs[:, pos]
    md_pinv = np.sum((proj_pos * proj_pos) / evals[pos][None, :], axis=1)

    d_diag = np.diag(np.diag(cov))
    cov_r = 0.90 * cov + 0.10 * d_diag + 1e-6 * np.eye(D)
    cov_r = 0.5 * (cov_r + cov_r.T)
    er, vr = np.linalg.eigh(cov_r)
    proj_r = Xc @ vr
    md_ridge = np.sum((proj_r * proj_r) / np.maximum(er[None, :], 1e-12), axis=1)

    k5 = min(5, N - 1, D - 1)
    V5 = evecs[:, -k5:]
    rec5 = (Xc @ V5) @ V5.T
    pca_k5 = ((Xc - rec5) ** 2).mean(axis=1)

    km = min(max(m_dim, 2), N - 1, D - 1)
    Vm = evecs[:, -km:]
    recm = (Xc @ Vm) @ Vm.T
    pca_km = ((Xc - recm) ** 2).mean(axis=1)

    p_col = np.clip(X.mean(axis=0), 1e-6, None)
    Z_mca = (X - p_col[None, :]) / np.sqrt(p_col[None, :])
    gram_mca = Z_mca.T @ Z_mca
    gram_mca = 0.5 * (gram_mca + gram_mca.T)
    _, evecs_mca = np.linalg.eigh(gram_mca)
    Vm5 = evecs_mca[:, -k5:]
    Z_hat = (Z_mca @ Vm5) @ Vm5.T
    mca_k5 = ((Z_mca - Z_hat) ** 2).mean(axis=1)

    return {
        "MD_1hot_pinv": md_pinv,
        "MD_1hot_ridge": md_ridge,
        "PCA_1hot_k5": pca_k5,
        "PCA_1hot_km": pca_km,
        "MCA_k5": mca_k5,
    }


def run_block_2_classical_and_pc1(results: dict):
    t0 = time.time()
    print("\n" + "=" * 78)
    print("BLOCK 2 [SNAP 2.2a & 2.2b]: Full-Feature 1-Hot MD/PCA/MCA + PC1 Sign-Aligned PT/EO")
    print("=" * 78, flush=True)
    full_feat = {}
    pc1_align = {}

    for ds in DS_8 + ["sadc_2017"]:
        cite = CITE[ds]
        data_df, y, bat = load_dataset_bundle(ds)
        _, vectorized, _, _ = prepare_for_model(data_df, {})
        X = vectorized.to_numpy(dtype=np.float64)
        m_dim = 2 if ds == "sadc_2017" else int(yaml.safe_load(open(f"cache/_tuned_{ds}_ae/best_hyperparameters.yaml"))["latent_space_dim"])
        sc_dict = onehot_classical_scores(X, m_dim)

        row_ff = {"N": int(X.shape[0]), "V": int(prepare_for_categorical(data_df.copy()).shape[1]), "D_onehot": int(X.shape[1])}
        for kname, s in sc_dict.items():
            nat, ori = auc_pair(y, s)
            row_ff[f"{kname}_native"] = nat
            row_ff[f"{kname}_oriented"] = ori
        full_feat[cite] = row_ff
        print(
            f"  [1-Hot] {cite:16s} (N={row_ff['N']:5d}, V={row_ff['V']:2d}, D={row_ff['D_onehot']:3d}): "
            f"MD_pinv={row_ff['MD_1hot_pinv_native']:.3f} MD_ridge={row_ff['MD_1hot_ridge_native']:.3f} "
            f"PCA5={row_ff['PCA_1hot_k5_native']:.3f} PCAm={row_ff['PCA_1hot_km_native']:.3f} MCA5={row_ff['MCA_k5_native']:.3f}",
            flush=True,
        )

        # PC1 sign alignment on ordinal battery (all 8 surveys with a battery!)
        if bat is not None and bat.shape[1] >= 2:
            B = bat.copy().astype(float)
            ls_raw = bl.longstring(B.copy())
            pt_raw = bl.person_total_r(B.copy())
            md_raw = bl.mahalanobis(B.copy())
            eo_raw = bl.even_odd(B.copy())
            irv_raw = bl.irv(B.copy())
            lz_raw = bl.lz(B.copy())

            B_imp = B.apply(lambda c: c.fillna(c.mean()), axis=0).to_numpy(dtype=float)
            sd = np.where(B_imp.std(axis=0) > 1e-8, B_imp.std(axis=0), 1.0)
            Z = (B_imp - B_imp.mean(axis=0)) / sd
            gz = (Z.T @ Z) / max(len(Z) - 1, 1)
            _, evz = np.linalg.eigh(0.5 * (gz + gz.T))
            w = evz[:, -1].copy()
            if (w >= 0).sum() < len(w) / 2.0:
                w = -w
            B_flip = B.copy()
            flipped_cols = []
            for j, col in enumerate(B.columns):
                if w[j] < 0:
                    cmin, cmax = np.nanmin(B[col]), np.nanmax(B[col])
                    B_flip[col] = cmax + cmin - B[col]
                    flipped_cols.append(col)
            pt_pc1 = bl.person_total_r(B_flip.copy())
            eo_pc1 = bl.even_odd(B_flip.copy())

            def _eval_idx(arr):
                m = np.isfinite(arr)
                if m.sum() < 10 or not (0 < y[m].sum() < m.sum()):
                    return None, None, int(m.sum())
                nat = float(roc_auc_score(y[m], arr[m]))
                return round(nat, 4), round(max(nat, 1.0 - nat), 4), int(m.sum())

            ls_nat, _ls_ori, _ = _eval_idx(ls_raw)
            pt_r_nat, pt_r_ori, n_pt = _eval_idx(pt_raw)
            pt_f_nat, pt_f_ori, _ = _eval_idx(pt_pc1)
            md_nat, _md_ori, _ = _eval_idx(md_raw)
            eo_r_nat, eo_r_ori, n_eo = _eval_idx(eo_raw)
            eo_f_nat, eo_f_ori, _ = _eval_idx(eo_pc1)
            irv_nat, _irv_ori, _ = _eval_idx(irv_raw)
            lz_nat, _lz_ori, _ = _eval_idx(lz_raw)

            pc1_align[cite] = {
                "J_battery": int(B.shape[1]),
                "n_flipped": len(flipped_cols),
                "n_valid_PT": n_pt,
                "n_valid_EO": n_eo,
                "LS_native": ls_nat,
                "IRV_native": irv_nat,
                "PT_raw_native": pt_r_nat,
                "PT_raw_oriented": pt_r_ori,
                "PT_pc1_native": pt_f_nat,
                "PT_pc1_oriented": pt_f_ori,
                "MD_ord_native": md_nat,
                "EO_raw_native": eo_r_nat,
                "EO_raw_oriented": eo_r_ori,
                "EO_pc1_native": eo_f_nat,
                "EO_pc1_oriented": eo_f_ori,
                "lz_native": lz_nat,
            }
            print(
                f"  [PC1  ] {cite:16s} (J={B.shape[1]:2d}, flipped={len(flipped_cols):2d}): "
                f"PT native {pt_r_nat} -> {pt_f_nat} | EO native {eo_r_nat} -> {eo_f_nat} (n_EO={n_eo})",
                flush=True,
            )

    keys9 = list(full_feat.keys())
    ff_means = {
        col: round(float(np.mean([full_feat[k][col] for k in keys9])), 4)
        for col in [
            "MD_1hot_pinv_native",
            "MD_1hot_pinv_oriented",
            "MD_1hot_ridge_native",
            "MD_1hot_ridge_oriented",
            "PCA_1hot_k5_native",
            "PCA_1hot_k5_oriented",
            "PCA_1hot_km_native",
            "PCA_1hot_km_oriented",
            "MCA_k5_native",
            "MCA_k5_oriented",
        ]
    }
    # Summary means across the 7 full-sample EO surveys (excluding mastroianni2022 where n_EO=36) and 8 PT surveys
    pt8_keys = [k for k, v in pc1_align.items() if v["PT_raw_native"] is not None]
    eo7_keys = [k for k, v in pc1_align.items() if v["EO_raw_native"] is not None and v["n_valid_EO"] >= 100]
    pc1_means = {
        "PT_8surveys_raw_native": round(float(np.mean([pc1_align[k]["PT_raw_native"] for k in pt8_keys])), 4),
        "PT_8surveys_pc1_native": round(float(np.mean([pc1_align[k]["PT_pc1_native"] for k in pt8_keys])), 4),
        "EO_7surveys_raw_native": round(float(np.mean([pc1_align[k]["EO_raw_native"] for k in eo7_keys])), 4),
        "EO_7surveys_pc1_native": round(float(np.mean([pc1_align[k]["EO_pc1_native"] for k in eo7_keys])), 4),
    }
    print(f"  9-SURVEY 1-HOT MEANS ({time.time()-t0:.1f}s):", ff_means, flush=True)
    print("  PC1 ALIGNMENT MEANS:", pc1_means, flush=True)
    results["full_feature_baselines"] = {"per_dataset": full_feat, "means_9": ff_means}
    results["pc1_sign_alignment"] = {"per_dataset": pc1_align, "means": pc1_means}
    save_ckpt(results)


def fit_cl_and_ae_on_subset(ds: str, df_sub: pd.DataFrame, seeds=(1, 2, 3, 4, 5)):
    cat = prepare_for_categorical(df_sub.copy()).reset_index(drop=True)
    cl = CLTree(alpha=1.0).fit(cat, random_state=2)
    cl_sc = (1.0 - cl.score_dataframe(cat)["pct"]).to_numpy()

    best = yaml.safe_load(open(f"cache/_tuned_{ds}_ae/best_hyperparameters.yaml"))
    bs = training_batch_size(ds)
    cfg = dict(best)
    cfg.update(epochs=120, percentile=85)
    ae_seed_scores = fast_keras_ae_seeds(cat, cfg, batch_size=bs, seeds=seeds)
    ranks = [stats.rankdata(e) / len(e) for e in ae_seed_scores]
    ae_ens = np.mean(ranks, axis=0)
    return cat.shape[1], cl_sc, ae_ens, ae_seed_scores


def curve_at_fracs(
    effect_fn,
    score: np.ndarray,
    fracs=(0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30),
):
    n = len(score)
    order = np.argsort(-score)
    out = {}
    for f in fracs:
        k = int(round(f * n))
        keep = np.setdiff1d(np.arange(n), order[:k], assume_unique=False)
        out[f"{int(round(f*100))}pct"] = round(float(effect_fn(keep)), 4)
    return out


def random_band(effect_fn, n: int, fracs=(0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30), B=200):
    rng = np.random.RandomState(42)
    out = {}
    for f in fracs:
        k = int(round(f * n))
        if k == 0:
            v = float(effect_fn(np.arange(n)))
            out[f"{int(round(f*100))}pct"] = {"mean": round(v, 4), "lo": round(v, 4), "hi": round(v, 4)}
            continue
        vals = [float(effect_fn(rng.choice(n, size=n - k, replace=False))) for _ in range(B)]
        out[f"{int(round(f*100))}pct"] = {
            "mean": round(float(np.mean(vals)), 4),
            "lo": round(float(np.percentile(vals, 5)), 4),
            "hi": round(float(np.percentile(vals, 95)), 4),
        }
    return out


def run_block_3_disjoint_downstream(results: dict):
    t0 = time.time()
    print("\n" + "=" * 78)
    print("BLOCK 3 [SNAP 2.3]: Out-of-Outcome ('Disjoint-Item') Downstream Substantive Cleaning")
    print("=" * 78, flush=True)
    disjoint_out = {}

    # ---------------- 1. ALVAREZ ET AL. (2019) [inattentive] ----------------
    ds = "inattentive"
    araw, y_alv, _ = load_dataset_bundle(ds)
    n_alv = len(araw)
    pol = [
        "support.Obamacare",
        "support.RepealDADT",
        "support.PathLegal",
        "support.CarbonLimits",
        "support.GunControl",
        "support.LimitSurveillance",
    ]
    mp = {"Support": 1.0, "Oppose": -1.0, "Indifferent": 0.0, "Don't know": 0.0}
    P = araw[pol].apply(lambda s: s.map(mp))

    def alv_constraint(idx):
        c = P.iloc[idx].corr().to_numpy()
        iu = np.triu_indices_from(c, 1)
        return float(np.nanmean(c[iu]))

    # Check ALL checks on Alvarez (primary `filter` vs union of checks!)
    lab_alv = aligned_labels(ds)
    y_alv_union = np.zeros(n_alv, dtype=bool)
    for ch in CHECKS[ds]:
        y_alv_union |= _relevant(lab_alv, ch).astype(bool)
    passers_alv_prim = np.where(y_alv == 0)[0]
    passers_alv_union = np.where(~y_alv_union)[0]

    cl_in = (1.0 - pd.read_csv(f"cache/_fix9_{ds}_cl/errors.csv")["pct"]).to_numpy()
    ae_in = pd.read_csv(f"cache/_tuned_{ds}_ae_p85/errors.csv")["error"].to_numpy()

    df_alv_31 = araw.drop(columns=pol)
    v31, cl_alv_31, ae_alv_31, ae_alv_31_seeds = fit_cl_and_ae_on_subset(ds, df_alv_31)

    derived_alv = [c for c in ["nonattitude_prop", "straightlining", "intransitive"] if c in df_alv_31.columns]
    df_alv_28 = df_alv_31.drop(columns=derived_alv)
    v28, cl_alv_28, ae_alv_28, ae_alv_28_seeds = fit_cl_and_ae_on_subset(ds, df_alv_28)

    disjoint_out["alvarez2019"] = {
        "N": n_alv,
        "full_rbar": round(alv_constraint(np.arange(n_alv)), 4),
        "passers_primary_rbar": round(alv_constraint(passers_alv_prim), 4),
        "n_passers_primary": int(len(passers_alv_prim)),
        "passers_union_rbar": round(alv_constraint(passers_alv_union), 4),
        "n_passers_union": int(len(passers_alv_union)),
        "random_band": random_band(alv_constraint, n_alv),
        "in_sample_37vars": {
            "CL_AUC": round(L.auc_native(y_alv, cl_in), 4),
            "AE_AUC": round(L.auc_native(y_alv, ae_in), 4),
            "CL_curve": curve_at_fracs(alv_constraint, cl_in),
            "AE_curve": curve_at_fracs(alv_constraint, ae_in),
        },
        "disjoint_31vars_no_policy": {
            "V": v31,
            "CL_AUC": round(L.auc_native(y_alv, cl_alv_31), 4),
            "AE_AUC_med5": round(float(np.median([L.auc_native(y_alv, s) for s in ae_alv_31_seeds])), 4),
            "CL_curve": curve_at_fracs(alv_constraint, cl_alv_31),
            "AE_curve": curve_at_fracs(alv_constraint, ae_alv_31),
            "AE_seed1_curve": curve_at_fracs(alv_constraint, ae_alv_31_seeds[0]),
        },
        "disjoint_28vars_strict": {
            "V": v28,
            "CL_AUC": round(L.auc_native(y_alv, cl_alv_28), 4),
            "AE_AUC_med5": round(float(np.median([L.auc_native(y_alv, s) for s in ae_alv_28_seeds])), 4),
            "CL_curve": curve_at_fracs(alv_constraint, cl_alv_28),
            "AE_curve": curve_at_fracs(alv_constraint, ae_alv_28),
            "AE_seed1_curve": curve_at_fracs(alv_constraint, ae_alv_28_seeds[0]),
        },
    }
    print("  [Alvarez 2019] done:", {k: v for k, v in disjoint_out["alvarez2019"].items() if k != "random_band"}, flush=True)

    # ---------------- 2. PENNYCOOK ET AL. (2020) [pennycook_1] ----------------
    ds = "pennycook_1"
    praw, y_pen, _ = load_dataset_bundle(ds)
    n_pen = len(praw)
    real_cols = [c for c in praw.columns if c.startswith("Real1_")]
    fake_cols = [c for c in praw.columns if c.startswith("Fake1_")]
    R_pen = praw[real_cols].apply(pd.to_numeric, errors="coerce").mean(1)
    F_pen = praw[fake_cols].apply(pd.to_numeric, errors="coerce").mean(1)

    def pen_disc(idx):
        return float(R_pen.iloc[idx].mean() - F_pen.iloc[idx].mean())

    lab_pen = aligned_labels(ds)
    y_pen_union = np.zeros(n_pen, dtype=bool)
    for ch in CHECKS[ds]:
        y_pen_union |= _relevant(lab_pen, ch).astype(bool)
    y_pen_union = y_pen_union.astype(int)
    passers_pen_ac1 = np.where(y_pen == 0)[0]
    passers_pen_all = np.where(y_pen_union == 0)[0]

    cl_pen_in = (1.0 - pd.read_csv(f"cache/_fix9_{ds}_cl/errors.csv")["pct"]).to_numpy()
    ae_pen_in = pd.read_csv(f"cache/_tuned_{ds}_ae_p85/errors.csv")["error"].to_numpy()

    df_pen_62 = praw.drop(columns=real_cols + fake_cols)
    v62, cl_pen_62, ae_pen_62, ae_pen_62_seeds = fit_cl_and_ae_on_subset(ds, df_pen_62)

    disjoint_out["pennycook2020"] = {
        "N": n_pen,
        "full_disc": round(pen_disc(np.arange(n_pen)), 4),
        "passers_ac1_disc": round(pen_disc(passers_pen_ac1), 4),
        "n_passers_ac1": int(len(passers_pen_ac1)),
        "passers_union_disc": round(pen_disc(passers_pen_all), 4),
        "n_passers_union": int(len(passers_pen_all)),
        "random_band": random_band(pen_disc, n_pen),
        "in_sample_92vars": {
            "CL_AUC_ac1": round(L.auc_native(y_pen, cl_pen_in), 4),
            "CL_AUC_union": round(L.auc_native(y_pen_union, cl_pen_in), 4),
            "CL_curve": curve_at_fracs(pen_disc, cl_pen_in),
            "AE_curve": curve_at_fracs(pen_disc, ae_pen_in),
        },
        "disjoint_62vars_no_headlines": {
            "V": v62,
            "CL_AUC_ac1": round(L.auc_native(y_pen, cl_pen_62), 4),
            "CL_AUC_union": round(L.auc_native(y_pen_union, cl_pen_62), 4),
            "AE_AUC_ac1_med5": round(float(np.median([L.auc_native(y_pen, s) for s in ae_pen_62_seeds])), 4),
            "AE_AUC_union_med5": round(float(np.median([L.auc_native(y_pen_union, s) for s in ae_pen_62_seeds])), 4),
            "CL_curve": curve_at_fracs(pen_disc, cl_pen_62),
            "AE_curve": curve_at_fracs(pen_disc, ae_pen_62),
            "AE_seed1_curve": curve_at_fracs(pen_disc, ae_pen_62_seeds[0]),
        },
    }
    print("  [Pennycook 2020] done:", {k: v for k, v in disjoint_out["pennycook2020"].items() if k != "random_band"}, flush=True)

    # ---------------- 3. O'GRADY ET AL. (2019) [moral_data] ----------------
    ds = "moral_data"
    oraw, y_og, _ = load_dataset_bundle(ds)
    n_og = len(oraw)
    omap = {"bottom-extreme": 0, "low": 1, "normal": 2, "high": 3, "top-extreme": 4}
    indiv = oraw["individualizing_cat"].map(omap).to_numpy(float)
    beh = np.nanmean(
        np.column_stack([pd.to_numeric(oraw[b], errors="coerce") for b in ["return1", "stamp1", "mailed1"]]),
        axis=1,
    )

    def og_corr(idx):
        x, yy = indiv[idx], beh[idx]
        m = np.isfinite(x) & np.isfinite(yy)
        return float(np.corrcoef(x[m], yy[m])[0, 1]) if m.sum() > 20 else np.nan

    passers_og = np.where(y_og == 0)[0]
    cl_og_in = (1.0 - pd.read_csv(f"cache/_fix9_{ds}_cl/errors.csv")["pct"]).to_numpy()
    ae_og_in = pd.read_csv(f"cache/_tuned_{ds}_ae_p85/errors.csv")["error"].to_numpy()

    beh_cols = [c for c in ["return1", "stamp1", "mailed1", "days", "comply1"] if c in oraw.columns]
    cat_sum_cols = [c for c in oraw.columns if c.endswith("_cat") and c != "temp_cat"]
    df_og_nobeh = oraw.drop(columns=beh_cols)
    v_nobeh, cl_og_nobeh, ae_og_nobeh, ae_og_nobeh_seeds = fit_cl_and_ae_on_subset(ds, df_og_nobeh)

    df_og_rawonly = oraw.drop(columns=beh_cols + cat_sum_cols)
    v_raw, cl_og_raw, ae_og_raw, ae_og_raw_seeds = fit_cl_and_ae_on_subset(ds, df_og_rawonly)

    indiv_q_cols = []
    for c in [col for col in df_og_rawonly.columns if col.startswith("q")]:
        qn = pd.to_numeric(df_og_rawonly[c], errors="coerce").to_numpy(float)
        m = np.isfinite(qn) & np.isfinite(indiv)
        if m.sum() > 20 and abs(np.corrcoef(qn[m], indiv[m])[0, 1]) > 0.25:
            indiv_q_cols.append(c)
    df_og_strict = df_og_rawonly.drop(columns=indiv_q_cols)
    v_strict, cl_og_strict, ae_og_strict, ae_og_strict_seeds = fit_cl_and_ae_on_subset(ds, df_og_strict)

    disjoint_out["ogrady2019"] = {
        "N": n_og,
        "full_r": round(og_corr(np.arange(n_og)), 4),
        "passers_r": round(og_corr(passers_og), 4),
        "n_passers": int(len(passers_og)),
        "in_sample_72vars": {
            "CL_AUC": round(L.auc_native(y_og, cl_og_in), 4),
            "AE_AUC": round(L.auc_native(y_og, ae_og_in), 4),
            "CL_curve": curve_at_fracs(og_corr, cl_og_in),
            "AE_curve": curve_at_fracs(og_corr, ae_og_in),
        },
        "disjoint_67vars_no_behavior": {
            "V": v_nobeh,
            "CL_AUC": round(L.auc_native(y_og, cl_og_nobeh), 4),
            "AE_AUC_med5": round(float(np.median([L.auc_native(y_og, s) for s in ae_og_nobeh_seeds])), 4),
            "CL_curve": curve_at_fracs(og_corr, cl_og_nobeh),
            "AE_curve": curve_at_fracs(og_corr, ae_og_nobeh),
        },
        "disjoint_53vars_raw_survey_only": {
            "V": v_raw,
            "CL_AUC": round(L.auc_native(y_og, cl_og_raw), 4),
            "AE_AUC_med5": round(float(np.median([L.auc_native(y_og, s) for s in ae_og_raw_seeds])), 4),
            "CL_curve": curve_at_fracs(og_corr, cl_og_raw),
            "AE_curve": curve_at_fracs(og_corr, ae_og_raw),
        },
        "disjoint_strict_no_indiv_no_beh": {
            "V": v_strict,
            "dropped_indiv_q": indiv_q_cols,
            "CL_AUC": round(L.auc_native(y_og, cl_og_strict), 4),
            "AE_AUC_med5": round(float(np.median([L.auc_native(y_og, s) for s in ae_og_strict_seeds])), 4),
            "CL_curve": curve_at_fracs(og_corr, cl_og_strict),
            "AE_curve": curve_at_fracs(og_corr, ae_og_strict),
        },
    }
    print(f"  [O'Grady 2019] done ({time.time()-t0:.1f}s):", disjoint_out["ogrady2019"], flush=True)
    results["disjoint_downstream"] = disjoint_out
    save_ckpt(results)


def partial_corr_ols(m: np.ndarray, c: np.ndarray, Z_df: pd.DataFrame):
    X = pd.get_dummies(Z_df.fillna("missing").astype(str), drop_first=True).to_numpy(dtype=np.float64)
    X = np.column_stack([np.ones(len(m)), X])
    beta_m, _, _, _ = np.linalg.lstsq(X, m, rcond=None)
    beta_c, _, _, _ = np.linalg.lstsq(X, c, rcond=None)
    rm = m - X @ beta_m
    rc = c - X @ beta_c
    return float(np.corrcoef(rm, rc)[0, 1])


def stratified_auc(m: np.ndarray, c: np.ndarray, z_series: pd.Series):
    s = pd.Series(z_series)
    num = pd.to_numeric(s, errors="coerce")
    lvl = (
        pd.qcut(num, 4, duplicates="drop").astype(str)
        if num.notna().mean() > 0.9 and num.nunique() > 8
        else s.fillna("missing").astype(str)
    )
    tot_w = 0
    acc = 0.0
    for _, idx in pd.Series(np.arange(len(m))).groupby(lvl.values):
        ii = idx.to_numpy()
        yy = c[ii]
        if len(ii) >= 10 and 0 < yy.sum() < len(yy):
            a = roc_auc_score(yy, m[ii])
            acc += len(ii) * a
            tot_w += len(ii)
    return float(acc / tot_w) if tot_w > 0 else np.nan


def run_block_4_covariate_validity_and_fairness(results: dict):
    t0 = time.time()
    print("\n" + "=" * 78)
    print("BLOCK 4 [SNAP 2.4b & 3.4]: Covariate-Adjusted Validity & 92-Var Disjoint Fairness")
    print("=" * 78, flush=True)
    cov_val = {}

    # Prepare reconciled mastroianni2022 (`public_opinion`, N=1003) scores & demographics
    _keep_po, sub_raw_po, y_po = public_opinion_completer_mask()
    d_po, _, _ = load_dataset_bundle("public_opinion")
    cat_po = prepare_for_categorical(d_po.copy()).reset_index(drop=True)
    cl_po = (1.0 - CLTree(alpha=1.0).fit(cat_po, random_state=2).score_dataframe(cat_po)["pct"]).to_numpy()
    best_po = yaml.safe_load(open("cache/_tuned_public_opinion_ae/best_hyperparameters.yaml"))
    cfg_po85 = dict(best_po); cfg_po85.update(epochs=120, percentile=85)
    ae_po_seeds = fast_keras_ae_seeds(d_po, cfg_po85, batch_size=training_batch_size("public_opinion"))
    ae_po = np.mean([stats.rankdata(e) / len(e) for e in ae_po_seeds], axis=0)

    # Prepare 92-var disjoint YRBS (`sadc_2017`) scores
    cat92, y_sadc, _ = load_dataset_bundle("sadc_2017")
    cl92_sc = (1.0 - CLTree(alpha=1.0).fit(cat92, mi_subsample=10000, random_state=2).score_dataframe(cat92)["pct"]).to_numpy()
    cfg92 = {
        "encoder_layers": 1, "encoder_units_1": 128, "encoder_activation_1": "relu",
        "latent_space_dim": 2, "latent_activation": "relu",
        "decoder_layers": 1, "decoder_units_1": 128, "decoder_activation_1": "relu",
        "learning_rate": 1e-4, "epochs": 5, "percentile": 85,
    }
    ae92_sc = fast_keras_ae_seeds(cat92, cfg92, batch_size=256, seeds=(1,))[0]

    for ds in ["inattentive", "public_opinion", "pennycook_1"]:
        cite = CITE[ds]
        if ds == "public_opinion":
            ae, cl, n = ae_po, cl_po, len(y_po)
            y = y_po.astype(float)
            demo = sub_raw_po.copy()
            axes = dict(F.AXES[ds])
            if "language" in demo.columns:
                axes["language"] = "language"
        else:
            sc = L.load_scores(ds)
            ae, cl, n = sc["AE"], sc["CL"], sc["N"]
            y = L.primary_label(ds).astype(float)
            demo = F._demographics(ds)
            axes = dict(F.AXES[ds])

        cols = [col for col in axes.values() if col in demo.columns]
        Z_all = demo[cols].copy()

        r_ae_raw = float(np.corrcoef(ae, y)[0, 1])
        r_cl_raw = float(np.corrcoef(cl, y)[0, 1])
        auc_ae_raw = L.auc_native(y, ae)
        auc_cl_raw = L.auc_native(y, cl)
        r_ae_joint = partial_corr_ols(ae, y, Z_all)
        r_cl_joint = partial_corr_ols(cl, y, Z_all)

        per_axis = {}
        for ax_name, col in axes.items():
            if col not in demo.columns:
                continue
            r_ae_z = partial_corr_ols(ae, y, demo[[col]])
            r_cl_z = partial_corr_ols(cl, y, demo[[col]])
            sauc_ae = stratified_auc(ae, y, demo[col])
            sauc_cl = stratified_auc(cl, y, demo[col])
            per_axis[ax_name] = {
                "Corr_AE_given_Z": round(r_ae_z, 4),
                "Corr_CL_given_Z": round(r_cl_z, 4),
                "AUC_AE_given_Z": round(sauc_ae, 4) if np.isfinite(sauc_ae) else None,
                "AUC_CL_given_Z": round(sauc_cl, 4) if np.isfinite(sauc_cl) else None,
            }

        cov_val[cite] = {
            "N": n,
            "n_fail": int(y.sum()),
            "axes": list(per_axis.keys()),
            "unadjusted": {
                "Corr_AE": round(r_ae_raw, 4),
                "Corr_CL": round(r_cl_raw, 4),
                "AUC_AE": round(auc_ae_raw, 4),
                "AUC_CL": round(auc_cl_raw, 4),
            },
            "joint_all_Z": {
                "Corr_AE_given_allZ": round(r_ae_joint, 4),
                "Corr_CL_given_allZ": round(r_cl_joint, 4),
            },
            "per_axis": per_axis,
        }
        print(
            f"  [{cite}] (N={n}, fail={int(y.sum())}) unadj Corr(AE/CL)={r_ae_raw:.3f}/{r_cl_raw:.3f} -> joint|Z={r_ae_joint:.3f}/{r_cl_joint:.3f}",
            flush=True,
        )
        for k, v in per_axis.items():
            print(
                f"    axis={k:12s}: Corr(AE/CL|Z)={v['Corr_AE_given_Z']:.3f}/{v['Corr_CL_given_Z']:.3f}  AUC(AE/CL|Z)={v['AUC_AE_given_Z']}/{v['AUC_CL_given_Z']}",
                flush=True,
            )

    # Commensurate binary-flag disparity across all 22 axes (16 well-powered: alvarez2019 + kann2018 + mastroianni2022)
    well_pow_set = {"inattentive", "sadc_2017", "public_opinion"}
    comm_rows = []
    for ds in ["inattentive", "sadc_2017", "public_opinion", "pennycook_1"]:
        cite = CITE[ds]
        if ds == "sadc_2017":
            ae_f, cl_f, C, demo = ae92_sc, cl92_sc, y_sadc, F._demographics("sadc_2017")
        elif ds == "public_opinion":
            ae_f, cl_f, C, demo = ae_po_seeds[0], cl_po, y_po, sub_raw_po
        else:
            ae_f, cl_f, n = F._flags(ds)
            C = F._check_fail(ds, n)
            demo = F._demographics(ds)
        passer = C == 0
        fr = float(C.mean())
        aez_p, clz_p = F._z(ae_f[passer]), F._z(cl_f[passer])
        for label, col in F.AXES[ds].items():
            if col not in demo.columns:
                continue
            Zall = demo[col].to_numpy()
            DC = L.eta(Zall, C, bins=4)
            if DC is None:
                continue
            DMc_ae = L.eta(Zall[passer], aez_p, bins=4)
            DMc_cl = L.eta(Zall[passer], clz_p, bins=4)
            DMf_ae = L.flag_disparity(ae_f, Zall, flag_rate=fr, bins=4)
            DMf_cl = L.flag_disparity(cl_f, Zall, flag_rate=fr, bins=4)
            comm_rows.append(
                {
                    "dataset": cite,
                    "axis": label,
                    "well_powered": ds in well_pow_set,
                    "flag_rate": round(fr, 4),
                    "D_C": round(DC, 3),
                    "DM_cont_AE": round(DMc_ae, 3) if DMc_ae is not None else None,
                    "DM_cont_CL": round(DMc_cl, 3) if DMc_cl is not None else None,
                    "DM_flag_AE": round(DMf_ae, 3) if DMf_ae is not None else None,
                    "DM_flag_CL": round(DMf_cl, 3) if DMf_cl is not None else None,
                    "AE_cont_fair": bool(DMc_ae is not None and DMc_ae <= DC),
                    "CL_cont_fair": bool(DMc_cl is not None and DMc_cl <= DC),
                    "AE_flag_fair": bool(DMf_ae is not None and DMf_ae <= DC),
                    "CL_flag_fair": bool(DMf_cl is not None and DMf_cl <= DC),
                }
            )
    cdf = pd.DataFrame(comm_rows)
    print(f"\n  [Commensurate Binary Flag Disparity ({time.time()-t0:.1f}s)]:")
    print(cdf.to_string(index=False), flush=True)
    results["covariate_adjusted_validity"] = cov_val
    results["commensurate_fairness"] = {
        "rows": comm_rows,
        "summary_all22": {
            "AE_cont_fair": int(cdf.AE_cont_fair.sum()),
            "CL_cont_fair": int(cdf.CL_cont_fair.sum()),
            "AE_flag_fair": int(cdf.AE_flag_fair.sum()),
            "CL_flag_fair": int(cdf.CL_flag_fair.sum()),
            "total": len(cdf),
        },
        "summary_well_powered_16": {
            "AE_cont_fair": int(cdf[cdf.well_powered].AE_cont_fair.sum()),
            "CL_cont_fair": int(cdf[cdf.well_powered].CL_cont_fair.sum()),
            "AE_flag_fair": int(cdf[cdf.well_powered].AE_flag_fair.sum()),
            "CL_flag_fair": int(cdf[cdf.well_powered].CL_flag_fair.sum()),
            "total": int(cdf.well_powered.sum()),
        },
    }
    save_ckpt(results)


def run_block_5_sample_size_and_coherence(results: dict):
    t0 = time.time()
    print("\n" + "=" * 78)
    print("BLOCK 5 [SNAP 3.3 & Coherence]: Sample-Size Fixed-Cohort/Native + 9-Survey Coherence")
    print("=" * 78, flush=True)
    df = pd.read_csv("sample_size_results.csv")
    m = df.groupby(["dataset", "N"])[["AE_AUC", "CL_AUC"]].mean().reset_index()
    grid = [100, 250, 500, 1000, 2500]

    def curve_summary(sub_m, col):
        out_pct, out_raw, out_se, counts = {}, {}, {}, {}
        for n in grid:
            vals_pct, vals_raw = [], []
            for _ds, g in sub_m.groupby("dataset"):
                fullN = g["N"].max()
                full = g.loc[g["N"] == fullN, col].iloc[0]
                if n <= fullN:
                    s = g.loc[g["N"] == n, col]
                    if len(s) and np.isfinite(full) and full > 0:
                        vals_pct.append(100.0 * s.iloc[0] / full)
                        vals_raw.append(float(s.iloc[0]))
            if vals_pct:
                out_pct[str(n)] = round(float(np.mean(vals_pct)), 2)
                out_raw[str(n)] = round(float(np.mean(vals_raw)), 4)
                out_se[str(n)] = round(float(np.std(vals_pct, ddof=1) / np.sqrt(len(vals_pct))) if len(vals_pct) > 1 else 0.0, 2)
                counts[str(n)] = len(vals_pct)
        return {"pct_of_full": out_pct, "se_pct": out_se, "raw_mean_auc": out_raw, "n_surveys": counts}

    ds_ge1000 = [ds for ds, g in m.groupby("dataset") if g["N"].max() >= 1000]
    m_fixed5 = m[m["dataset"].isin(ds_ge1000)]

    # Native-orientation CL sample-size check across all 9 reconciled surveys
    from experiment_b.sample_size import _subsample

    cl_nat_rows = []
    for ds in DS_8 + ["sadc_2017"]:
        data_raw, y, _ = load_dataset_bundle(ds)
        data_df = prepare_for_categorical(data_raw.copy()).reset_index(drop=True)
        Ntot = len(data_df)
        for N in [n for n in grid if n < Ntot] + [Ntot]:
            for seed in [1, 2, 3]:
                idx = _subsample(y, N, seed) if N < Ntot else np.arange(Ntot)
                sub = data_df.iloc[idx].reset_index(drop=True)
                cl = CLTree(alpha=1.0).fit(sub, mi_subsample=(10000 if len(sub) > 12000 else None), random_state=2)
                sc = (1.0 - cl.score_dataframe(data_df)["pct"]).to_numpy()
                nat, ori = auc_pair(y, sc)
                cl_nat_rows.append(
                    {"dataset": ds, "N": int(len(idx)), "seed": seed, "CL_native": nat, "CL_oriented": ori}
                )
    cl_nat_df = pd.DataFrame(cl_nat_rows).groupby(["dataset", "N"])[["CL_native", "CL_oriented"]].mean().reset_index()

    sample_size_info = {
        "all_eligible": {"CL": curve_summary(m, "CL_AUC"), "AE": curve_summary(m, "AE_AUC")},
        "fixed_cohort_ge1000": {
            "datasets": [CITE[d] for d in ds_ge1000],
            "CL": curve_summary(m_fixed5, "CL_AUC"),
            "AE": curve_summary(m_fixed5, "AE_AUC"),
        },
        "cl_reconciled_native_vs_oriented": {
            "native_all": curve_summary(cl_nat_df, "CL_native"),
            "oriented_all": curve_summary(cl_nat_df, "CL_oriented"),
            "native_fixed5": curve_summary(cl_nat_df[cl_nat_df["dataset"].isin(ds_ge1000)], "CL_native"),
        },
    }
    print("  [Sample Size Summary]:", json.dumps(sample_size_info, indent=2), flush=True)

    # Empirical coherence across all 9 reconciled surveys
    coh_rows = []
    for ds in DS_8 + ["sadc_2017"]:
        cite = CITE[ds]
        data_raw, _, bat = load_dataset_bundle(ds)
        cat = prepare_for_categorical(data_raw.copy()).reset_index(drop=True)
        sub = cat.iloc[:2000] if len(cat) > 2000 else cat
        codes = [pd.Categorical(sub[c]).codes for c in sub.columns]
        ent = []
        for cd in codes:
            _, cnts = np.unique(cd, return_counts=True)
            p = cnts / cnts.sum()
            ent.append(float(-(p * np.log(p + 1e-12)).sum()))
        V = len(codes)
        nmi_mat = np.zeros((V, V), dtype=float)
        for i in range(V):
            for j in range(i + 1, V):
                mi = mutual_info_score(codes[i], codes[j])
                denom = 0.5 * (ent[i] + ent[j])
                val = (mi / denom) if denom > 1e-9 else 0.0
                nmi_mat[i, j] = val
                nmi_mat[j, i] = val
        iu = np.triu_indices(V, 1)
        mean_nmi = float(nmi_mat[iu].mean())
        max_nbr_nmi = float(nmi_mat.max(axis=1).mean())

        alpha_pc1 = np.nan
        if bat is not None and bat.shape[1] >= 2:
            B = bat.apply(pd.to_numeric, errors="coerce")
            B_imp = B.apply(lambda c: c.fillna(c.mean()), axis=0).to_numpy(float)
            sd = np.where(B_imp.std(0) > 1e-8, B_imp.std(0), 1.0)
            Z = (B_imp - B_imp.mean(0)) / sd
            gz = (Z.T @ Z) / max(len(Z) - 1, 1)
            _, evz = np.linalg.eigh(0.5 * (gz + gz.T))
            w = evz[:, -1].copy()
            if (w >= 0).sum() < len(w) / 2.0:
                w = -w
            B_al = B_imp.copy()
            for j in range(B_al.shape[1]):
                if w[j] < 0:
                    B_al[:, j] = B_al[:, j].max() + B_al[:, j].min() - B_al[:, j]
            k = B_al.shape[1]
            vt_sum = B_al.sum(1).var(ddof=1)
            if vt_sum > 0:
                alpha_pc1 = float((k / (k - 1.0)) * (1.0 - B_al.var(0, ddof=1).sum() / vt_sum))

        ab = results["ablation_2x2"]["per_dataset"][cite]
        coh_rows.append(
            {
                "dataset": cite,
                "V": V,
                "mean_pairwise_NMI": round(mean_nmi, 4),
                "mean_max_nbr_NMI": round(max_nbr_nmi, 4),
                "alpha_pc1": round(alpha_pc1, 4) if np.isfinite(alpha_pc1) else None,
                "CL_AUC": ab["CL_native"],
                "AE85_AUC": ab["AE_85_med_native"],
            }
        )
    coh_df = pd.DataFrame(coh_rows)
    print(f"\n  [9-Survey Coherence Table ({time.time()-t0:.1f}s)]:")
    print(coh_df.to_string(index=False), flush=True)
    coh_corrs = {}
    for xcol in ["mean_pairwise_NMI", "mean_max_nbr_NMI", "alpha_pc1"]:
        subc = coh_df.dropna(subset=[xcol])
        coh_corrs[xcol] = {}
        for ycol in ["CL_AUC", "AE85_AUC"]:
            pr = stats.pearsonr(subc[xcol], subc[ycol])
            sr = stats.spearmanr(subc[xcol], subc[ycol])
            coh_corrs[xcol][ycol] = {
                "n": int(len(subc)),
                "pearson_r": round(float(pr.statistic), 4),
                "pearson_p": round(float(pr.pvalue), 4),
                "spearman_rho": round(float(sr.statistic), 4),
                "spearman_p": round(float(sr.pvalue), 4),
            }
            print(
                f"    {xcol:20s} vs {ycol:8s} (n={len(subc)}): r={pr.statistic:+.3f} (p={pr.pvalue:.3f}), rho={sr.statistic:+.3f} (p={sr.pvalue:.3f})",
                flush=True,
            )

    results["sample_size_analysis"] = sample_size_info
    results["survey_coherence"] = {"rows": coh_rows, "correlations": coh_corrs}
    save_ckpt(results)


def main():
    t_all = time.time()
    resume = ("--resume" in sys.argv) or (os.environ.get("RESUME", "0") == "1")
    ckpt_src = CHECKPOINT_JSON or OUT_JSON
    results = json.load(open(ckpt_src)) if (resume and os.path.exists(ckpt_src)) else {}
    if "ablation_2x2" not in results:
        run_block_1_ablation(results)
    else:
        print(f"Skipping Block 1 (resumed from {ckpt_src})", flush=True)
    if "full_feature_baselines" not in results:
        run_block_2_classical_and_pc1(results)
    if "disjoint_downstream" not in results:
        run_block_3_disjoint_downstream(results)
    if "covariate_adjusted_validity" not in results:
        run_block_4_covariate_validity_and_fairness(results)
    if "sample_size_analysis" not in results:
        run_block_5_sample_size_and_coherence(results)
    os.makedirs(os.path.dirname(os.path.abspath(OUT_JSON)), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nALL 5 [SNAP] BLOCKS COMPLETED SUCCESSFULLY in {time.time()-t_all:.1f}s -> {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
