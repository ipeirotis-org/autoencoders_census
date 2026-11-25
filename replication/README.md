# Zenodo Replication Package Builder

This directory builds, verifies, and (optionally) stages a Zenodo deposit for:

> *"Learning to Pay Attention: Unsupervised Modeling of Attentive and Inattentive Survey Respondents"* (*MIS Quarterly*)

## 1. Build Both Archives Locally (Zero External Dependencies)

Ensure the paper evaluation harness branch is fetched if it is not yet merged into `main`:

```bash
git fetch origin revision/experiment-scripts
```

Then run from the repository root (`autoencoders_census/`):

```bash
# Camera-ready Zenodo bundle (full metadata: .zenodo.json, CITATION.cff, live demo link)
python3 replication/build_zenodo_package.py --mode camera-ready --version 1.0.0

# Double-anonymous peer-review bundle (stripped metadata + fail-closed text & PDF audit)
python3 replication/build_zenodo_package.py --mode blind --version 1.0.0
```

*(In `--mode blind`, the builder fails closed if `docs/MISQ_online_supplement.pdf` cannot be inspected for text and metadata leaks via `pdftotext`+`pdfinfo` or `pymupdf`; pass `--omit-supplement-pdf` on minimal hosts without PDF tooling.)*

Outputs are written to `dist/zenodo/` (gitignored):
- `inattentiveness-screening-replication-v1.0.0.zip` (recommended for Zenodo web preview)
- `inattentiveness-screening-replication-v1.0.0.tar.gz`
- `inattentiveness-screening-replication-v1.0.0.zenodo.json`
- `inattentiveness-screening-replication-blind-v1.0.0.zip` & `.tar.gz`

## 2. Hydrate Precomputed Score Caches from the `2026-10-23` Paper-Data Snapshot

On a machine with access to the restored `gs://autoencoders-census-paper-data/snapshots/2026-10-23` snapshot:

```bash
python3 replication/build_zenodo_package.py \
  --mode camera-ready \
  --version 1.0.0 \
  --snapshot-dir /path/to/restored/autoencoders_census
```

By default, `--snapshot-dir` copies **only derived, anonymous score caches and tuning configs** (`errors.csv`, `best_hyperparameters.yaml`, `oracle.json`, `metrics.csv`, `averages.csv`) and does **not** redistribute third-party raw survey CSVs or large TensorFlow SavedModel weight directories.
- Pass `--include-weights` if you also want TensorFlow SavedModel checkpoints bundled.
- Pass `--include-third-party-raw-data` **only** after verifying each external survey's license permits redistribution (see `DATA_PROVENANCE.md` inside the built archive).

## 3. Stage a Private Draft Deposit on Zenodo (Reserve DOI Without Publishing)

```bash
# Dry run on Zenodo Sandbox first
export ZENODO_TOKEN="<sandbox-personal-access-token>"
python3 replication/upload_to_zenodo.py \
  --package-dir dist/zenodo/inattentiveness-screening-replication-v1.0.0

# Production Zenodo DRAFT with reserved DOI (safe during double-blind review; stays private)
export ZENODO_TOKEN="<production-personal-access-token>"
python3 replication/upload_to_zenodo.py \
  --production \
  --package-dir dist/zenodo/inattentiveness-screening-replication-v1.0.0
```

This prints your **Deposit ID** and **Reserved DOI** and saves them to `dist/zenodo/inattentiveness-screening-replication-v1.0.0.deposit.json`.

Publish **that exact reserved draft** on production Zenodo **only upon acceptance** (or when instructed by the MISQ Research Transparency Editor), passing `--deposition-id <DEPOSITION_ID>` so the reserved DOI resolves to the published archive (rather than creating a duplicate deposit):

```bash
python3 replication/upload_to_zenodo.py \
  --production \
  --deposition-id <DEPOSITION_ID> \
  --publish \
  --yes-i-know-this-is-public-and-permanent \
  --package-dir dist/zenodo/inattentiveness-screening-replication-v1.0.0
```

## 4. MISQ Research Transparency Form

See [`MISQ_TRANSPARENCY_COMMITMENT_DRAFT.md`](MISQ_TRANSPARENCY_COMMITMENT_DRAFT.md) for copy-paste text to place into `templates/MISQ_Research_Transparency_Commitment.pdf`.
