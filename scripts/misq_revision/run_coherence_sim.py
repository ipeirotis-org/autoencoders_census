"""
Controlled Coherence Simulation (Section 7.5 / Referee Minor 3.1).

Evaluates unsupervised detectors on J-item, K=5-category scales (N=800,
85% attentive n_att=680, 15% careless n_inatt=120, 12 Monte Carlo replications):
  1. Chow-Liu tree (Laplace alpha=1.0) negative log-likelihood
  2. Percentile-Loss Autoencoder (p=85, 1 hidden layer 32 ReLU -> bottleneck m=5
     -> per-item 5-way softmax, variable-normalized cross-entropy, B=64)
  3. Rank-5 PCA squared Euclidean reconstruction error on centered one-hot matrix
  4. Inter-item response variability (IRV; negative within-person SD across items),
     highlighting complementarity on invariant straightlining vs. random responding.

Four experimental panels around baseline (J=12, lambda=0.60, 0% reverse-keyed):
  (a) Common-factor loading lambda in {0.15, 0.30, 0.45, 0.60, 0.75, 0.90} (random)
  (b) Battery breadth J in {4, 8, 12, 20, 32} (random)
  (c) Reverse-keyed share in {0%, 25%, 50%} (random)
  (d) Reverse-keyed share in {0%, 25%, 50%} (15% invariant straightlining)
"""
import json
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("REPO_ROOT", os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..")))
sys.path.insert(0, REPO)
from chow_liu_rank import CLTree

OUT_JSON = os.environ.get("OUT_JSON", os.path.join(SCRIPT_DIR, "results", "coherence_sim_results.json"))
PAPER_DIR = os.environ.get("PAPER_DIR", os.path.abspath(os.path.join(REPO, "..", "inattentiveness_paper")))
OUT_FIG = os.environ.get("OUT_FIG", os.path.join(PAPER_DIR, "figures", "coherence_simulation.pdf"))

N_TOTAL = 800
N_ATT = 680
N_INATT = 120
K_CATS = 5
N_SEEDS = 12


def simulate_survey(J, lam, rev_frac, careless_type, seed):
    rng = np.random.default_rng(seed)
    theta = rng.standard_normal(N_ATT)
    eps = rng.standard_normal((N_ATT, J))
    n_rev = int(round(J * rev_frac))
    signs = np.ones(J, dtype=float)
    if n_rev > 0:
        signs[:n_rev] = -1.0
    z = lam * np.outer(theta, signs) + np.sqrt(max(0.0, 1.0 - lam ** 2)) * eps

    X_att = np.zeros((N_ATT, J), dtype=int)
    for j in range(J):
        ranks = np.argsort(np.argsort(z[:, j]))
        X_att[:, j] = np.minimum((ranks * K_CATS) // N_ATT, K_CATS - 1)

    if careless_type == "uniform":
        X_inatt = rng.integers(0, K_CATS, size=(N_INATT, J))
    elif careless_type == "straightline":
        chosen = rng.integers(0, K_CATS, size=(N_INATT, 1))
        X_inatt = np.repeat(chosen, J, axis=1)
    else:
        raise ValueError(careless_type)

    X = np.vstack([X_att, X_inatt])
    y = np.concatenate([np.zeros(N_ATT, dtype=int), np.ones(N_INATT, dtype=int)])
    perm = rng.permutation(N_TOTAL)
    return X[perm], y[perm]


def one_hot_encode(X, J, K=K_CATS):
    N = X.shape[0]
    oh = np.zeros((N, J, K), dtype=np.float32)
    oh[np.arange(N)[:, None], np.arange(J)[None, :], X] = 1.0
    return oh.reshape(N, J * K)


def score_chow_liu(X, J):
    df = pd.DataFrame({f"v{j}": X[:, j].astype(str) for j in range(J)})
    cl = CLTree(alpha=1.0).fit(df, mi_subsample=None)
    return -cl.log_likelihood(df)


def score_pca(X_oh, k=5):
    X_centered = X_oh - X_oh.mean(axis=0, keepdims=True)
    n_comp = min(k, X_centered.shape[1] - 1)
    pca = PCA(n_components=n_comp, svd_solver="full", random_state=0)
    Z = pca.fit_transform(X_centered)
    X_hat = pca.inverse_transform(Z)
    return np.sum((X_centered - X_hat) ** 2, axis=1)


def score_irv(X):
    return -np.std(X.astype(float), axis=1, ddof=1)


class PLAutoencoderTrainer:
    """Compiled TF graph per item count J with explicit RNG Glorot re-initialization."""

    def __init__(self, J, K=K_CATS, hidden=32, bottleneck=5, p=85.0, lr=3e-3):
        self.J = J
        self.K = K
        self.D = J * K
        self.p = p
        self.hidden = hidden
        self.bottleneck = bottleneck
        self.lr = lr
        self._build()

    def _build(self):
        inp = keras.Input(shape=(self.D,))
        h = keras.layers.Dense(self.hidden, activation="relu")(inp)
        z = keras.layers.Dense(self.bottleneck, activation="relu")(h)
        dec = keras.layers.Dense(self.hidden, activation="relu")(z)
        logits = keras.layers.Dense(self.D)(dec)
        reshaped = keras.layers.Reshape((self.J, self.K))(logits)
        probs = keras.layers.Softmax(axis=-1)(reshaped)
        out = keras.layers.Reshape((self.D,))(probs)
        self.model = keras.Model(inp, out)
        self.opt = keras.optimizers.Adam(learning_rate=self.lr)
        # Force optimizer slot creation once
        dummy = tf.zeros((4, self.D), dtype=tf.float32)
        with tf.GradientTape() as tape:
            loss0 = tf.reduce_mean(self.model(dummy, training=True))
        grads0 = tape.gradient(loss0, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads0, self.model.trainable_variables))

        @tf.function
        def train_step(x_batch):
            with tf.GradientTape() as tape:
                y_pred = self.model(x_batch, training=True)
                y_true_r = tf.reshape(x_batch, (-1, self.J, self.K))
                y_pred_r = tf.clip_by_value(
                    tf.reshape(y_pred, (-1, self.J, self.K)), 1e-7, 1.0 - 1e-7
                )
                ce_per_var = -tf.reduce_sum(y_true_r * tf.math.log(y_pred_r), axis=-1) / np.log(self.K)
                sample_loss = tf.reduce_mean(ce_per_var, axis=-1)
                bsz = tf.shape(sample_loss)[0]
                k_keep = tf.maximum(
                    1, tf.cast(tf.math.ceil(tf.cast(bsz, tf.float32) * (self.p / 100.0)), tf.int32)
                )
                sorted_loss = tf.sort(sample_loss, direction="ASCENDING")
                loss = tf.reduce_mean(sorted_loss[:k_keep])
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
            return loss

        self._train_step = train_step

    def reset_weights(self, seed):
        rng = np.random.default_rng(seed)
        for layer in self.model.layers:
            if hasattr(layer, "kernel") and layer.kernel is not None:
                fin, fout = layer.kernel.shape
                # Per-variable output head uses fan_out = K_CATS
                eff_fout = self.K if fout == self.D else fout
                limit = np.sqrt(6.0 / (fin + eff_fout))
                w = rng.uniform(-limit, limit, size=(fin, fout)).astype(np.float32)
                layer.kernel.assign(w)
            if hasattr(layer, "bias") and layer.bias is not None:
                layer.bias.assign(np.zeros(layer.bias.shape, dtype=np.float32))
        for var in self.opt.variables:
            var.assign(tf.zeros_like(var))

    def fit_and_score(self, X_oh, seed, epochs=80, batch_size=64):
        self.reset_weights(seed)
        rng = np.random.default_rng(seed + 10000)
        N = X_oh.shape[0]
        for _ in range(epochs):
            idx = rng.permutation(N)
            X_shuf = X_oh[idx]
            for start in range(0, N, batch_size):
                self._train_step(tf.constant(X_shuf[start:start + batch_size]))
        preds = self.model(tf.constant(X_oh), training=False).numpy()
        y_true_r = X_oh.reshape(N, self.J, self.K)
        y_pred_r = np.clip(preds.reshape(N, self.J, self.K), 1e-7, 1.0 - 1e-7)
        ce_per_var = -np.sum(y_true_r * np.log(y_pred_r), axis=-1) / np.log(self.K)
        return np.mean(ce_per_var, axis=-1)


def main():
    t0 = time.time()
    trainers = {J: PLAutoencoderTrainer(J=J) for J in [4, 8, 12, 20, 32]}

    conditions = []
    for lam in [0.15, 0.30, 0.45, 0.60, 0.75, 0.90]:
        conditions.append(("panel_a", lam, 12, lam, 0.0, "uniform"))
    for J in [4, 8, 12, 20, 32]:
        conditions.append(("panel_b", J, J, 0.60, 0.0, "uniform"))
    for rev in [0.0, 0.25, 0.50]:
        conditions.append(("panel_c", int(rev * 100), 12, 0.60, rev, "uniform"))
    for rev in [0.0, 0.25, 0.50]:
        conditions.append(("panel_d", int(rev * 100), 12, 0.60, rev, "straightline"))

    cache = {}
    results = {"panel_a": [], "panel_b": [], "panel_c": [], "panel_d": []}

    for panel, xval, J, lam, rev, ctype in conditions:
        key = (J, round(lam, 4), round(rev, 4), ctype)
        if key not in cache:
            cl_aucs, ae_aucs, pca_aucs, pca1_aucs, irv_aucs = [], [], [], [], []
            for s in range(N_SEEDS):
                X, y = simulate_survey(J, lam, rev, ctype, seed=1000 + s)
                X_oh = one_hot_encode(X, J)
                s_cl = score_chow_liu(X, J)
                s_pca = score_pca(X_oh, k=5)
                s_pca1 = score_pca(X_oh, k=1)
                s_irv = score_irv(X)
                s_ae = trainers[J].fit_and_score(X_oh, seed=2000 + s * 17, epochs=80, batch_size=64)
                cl_aucs.append(float(roc_auc_score(y, s_cl)))
                pca_aucs.append(float(roc_auc_score(y, s_pca)))
                pca1_aucs.append(float(roc_auc_score(y, s_pca1)))
                irv_aucs.append(float(roc_auc_score(y, s_irv)))
                ae_aucs.append(float(roc_auc_score(y, s_ae)))
            cache[key] = {
                "cl_mean": float(np.mean(cl_aucs)), "cl_sd": float(np.std(cl_aucs, ddof=1)),
                "ae_mean": float(np.mean(ae_aucs)), "ae_sd": float(np.std(ae_aucs, ddof=1)),
                "pca_mean": float(np.mean(pca_aucs)), "pca_sd": float(np.std(pca_aucs, ddof=1)),
                "pca1_mean": float(np.mean(pca1_aucs)), "pca1_sd": float(np.std(pca1_aucs, ddof=1)),
                "irv_mean": float(np.mean(irv_aucs)), "irv_sd": float(np.std(irv_aucs, ddof=1)),
            }
            print(
                f"{key}: CL={cache[key]['cl_mean']:.3f}+-{cache[key]['cl_sd']:.3f} | "
                f"AE={cache[key]['ae_mean']:.3f}+-{cache[key]['ae_sd']:.3f} | "
                f"PCA5={cache[key]['pca_mean']:.3f}+-{cache[key]['pca_sd']:.3f} | "
                f"PCA1={cache[key]['pca1_mean']:.3f}+-{cache[key]['pca1_sd']:.3f} | "
                f"IRV={cache[key]['irv_mean']:.3f}+-{cache[key]['irv_sd']:.3f}"
            )
        results[panel].append({"x": xval, **cache[key]})

    results["elapsed_sec"] = round(time.time() - t0, 2)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {OUT_JSON} in {results['elapsed_sec']}s")

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8.8,
        "axes.titlesize": 9.2,
        "axes.labelsize": 8.8,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 4, figsize=(11.4, 2.85), sharey=True)
    specs = [
        ("panel_a", r"(a) Factor loading ($J=12$, random)", r"Common-factor loading $\lambda$",
         [0.15, 0.30, 0.45, 0.60, 0.75, 0.90], None),
        ("panel_b", r"(b) Battery breadth ($\lambda=0.60$, random)", r"Number of items $J$",
         [4, 8, 12, 20, 32], None),
        ("panel_c", r"(c) Reverse-keying (random)", r"Reverse-keyed items (%)",
         [0, 25, 50], ["0%", "25%", "50%"]),
        ("panel_d", r"(d) Reverse-keying (straightlining)", r"Reverse-keyed items (%)",
         [0, 25, 50], ["0%", "25%", "50%"]),
    ]
    series_cfg = [
        ("cl", "Chow-Liu", "#D55E00", "o", "-"),
        ("ae", "PL-AE ($p=85$)", "#009E73", "^", "-"),
        ("pca", "PCA ($k=5$)", "#0072B2", "s", "--"),
        ("irv", "IRV (dispersion)", "#7F7F7F", "d", ":"),
    ]

    for _idx_ax, (ax, (pkey, title, xlabel, xticks, xticklabels)) in enumerate(zip(axes, specs)):
        rows = results[pkey]
        xs = [r["x"] for r in rows]
        ax.axhline(0.50, color="#999999", linestyle="--", linewidth=0.9, zorder=1)
        for prefix, label, color, marker, ls in series_cfg:
            # Show IRV on panels (c) and (d) or all panels for complete transparency
            means = np.array([r[f"{prefix}_mean"] for r in rows])
            sds = np.array([r[f"{prefix}_sd"] for r in rows])
            ax.fill_between(xs, means - sds, means + sds, color=color, alpha=0.13, linewidth=0)
            ax.plot(
                xs, means, label=label, color=color, marker=marker,
                linestyle=ls, linewidth=1.55, markersize=4.5, zorder=3
            )
        ax.set_title(title, pad=6)
        ax.set_xlabel(xlabel)
        ax.set_xticks(xticks)
        if xticklabels is not None:
            ax.set_xticklabels(xticklabels)
        ax.set_ylim(-0.04, 1.04)
        ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
        ax.grid(True, axis="y", color="#EAEAEA", linewidth=0.6, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Detection AUC")
    axes[1].legend(
        loc="lower right", frameon=True, facecolor="white",
        framealpha=0.92, edgecolor="#CCCCCC", handlelength=1.8
    )

    fig.tight_layout(w_pad=0.9)
    fig.savefig(OUT_FIG, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {OUT_FIG}")


if __name__ == "__main__":
    main()
