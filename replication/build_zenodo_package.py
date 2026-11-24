#!/usr/bin/env python3
"""Build a verified, upload-ready Zenodo replication package for:

    "Learning to Pay Attention: Unsupervised Modeling of Attentive and
     Inattentive Survey Respondents" (MIS Quarterly submission)

Supports two modes:
  - camera-ready (default): Full Zenodo deposit bundle with author/affiliation
    metadata (.zenodo.json, CITATION.cff), companion web demo link, GitHub
    references, Lean 4 formalization, verified revision results, and the
    unreviewed technical supplement PDF.
  - blind: Anonymized peer-review archive with all author names, institutional
    affiliations, GitHub organization handles, and live-demo URLs stripped and
    verified by an automated zero-leak scan.

Optionally hydrates precomputed per-respondent score caches (and, if explicitly
requested, third-party benchmark CSVs) from a restored paper-data snapshot via
--snapshot-dir.

Zero external dependencies: runs on standard-library Python 3.10+.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import py_compile
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PAPER_TITLE = (
    "Learning to Pay Attention: Unsupervised Modeling of Attentive and "
    "Inattentive Survey Respondents"
)

# Files/directories copied from the code repo working tree (main branch).
MAIN_CODE_FILES = [
    "main.py",
    "chow_liu_rank.py",
    "afv_lift.py",
    "run_stats.py",
    "benchmark_ae_vs_cl.py",
    "make_plots.py",
    "ruff.toml",
]

MAIN_CODE_DIRS = [
    "features",
    "train",
    "config",
    "scripts/misq_revision",
]

# Scientific / ML unit & integration tests (excludes GCP/PubSub/Firestore worker tests
# so `pytest tests/ -v` passes cleanly under minimal `requirements-replication.txt`).
PURE_ML_TESTS = [
    "tests/__init__.py",
    "tests/dataset/__init__.py",
    "tests/dataset/test_loader.py",
    "tests/features/__init__.py",
    "tests/features/test_data_leakage.py",
    "tests/features/test_transform.py",
    "tests/model/__init__.py",
    "tests/model/test_autoencoder.py",
    "tests/model/test_loss.py",
    "tests/test_chow_liu.py",
    "tests/test_contributions.py",
    "tests/test_integration_pipeline.py",
    "tests/test_per_column_errors.py",
]

# Core Python packages where revision/experiment-scripts carries the exact
# paper evaluation harness, binning fix (>=20 unique values), and Experiment B /
# revision scripts (not yet merged into main as of PR #59).
OVERLAY_PATHS = [
    "dataset/__init__.py",
    "dataset/loader.py",
    "model/__init__.py",
    "model/autoencoder.py",
    "model/base.py",
    "model/factory.py",
    "model/gan.py",
    "model/layers.py",
    "model/loss.py",
    "model/pcamodel.py",
    "model/presets.py",
    "model/variational_autoencoder.py",
    "evaluate/__init__.py",
    "evaluate/baselines.py",
    "evaluate/detection.py",
    "evaluate/evaluator.py",
    "evaluate/generator.py",
    "evaluate/outliers.py",
    "utils.py",
    "config/hp_autoencoder_real.yaml",
    "config/linear_autoencoder.yaml",
    "tests/dataset/test_loader.py",
    "experiment_b",
    "revision_experiments",
]

# Public-domain datasets tracked directly in autoencoders_census/data/
PUBLIC_DOMAIN_DATA_FILES = [
    "data/sadc_2017only_national_full.csv",
]

# Non-YRBS benchmark CSV filenames expected under code/data/
EXTERNAL_BENCHMARK_CSVS = [
    "Pennycook et al._Study 1.csv",
    "inattentive_users.csv",
    "attention_check.csv",
    "moral_data.csv",
    "Bot_Bot_Bot__MTURK.csv",
    "ethics.csv",
    "public_opinion.csv",
    "racial_data.csv",
]

# Score-cache patterns eligible for hydration from --snapshot-dir/cache
SCORE_CACHE_ALLOW_FILENAMES = {
    "errors.csv",
    "best_hyperparameters.yaml",
    "oracle.json",
    "metrics.csv",
    "averages.csv",
}

# Derived root-level experiment CSVs (aggregate metrics only, no raw survey rows)
# eligible for hydration from --snapshot-dir into code/.
DERIVED_SNAPSHOT_CSVS = [
    "sample_size_results.csv",
    "pl_sweep_results.csv",
    "linear_ae_lift.csv",
    "reconstruction_table.csv",
]

SKIP_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".git",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    ".DS_Store",
}

SKIP_EXTENSIONS = {".pyc", ".pyo", ".pyd"}

# Patterns that must NEVER appear in any staged text file.
SECRET_PATTERNS = [
    (re.compile(r"/Users/[A-Za-z0-9._-]+/"), "hardcoded macOS user path"),
    (re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY"), "private key block"),
    (re.compile(r"service[-_]account[-_]key\.json", re.I), "GCP service account key ref"),
    (re.compile(r"AIza[0-9A-Za-z\-_]{35}"), "Google API key"),
]

# Patterns that must NEVER appear in --mode blind packages.
BLIND_FORBIDDEN_PATTERNS = [
    (re.compile(r"\bipeirotis\b", re.I), "author surname (Ipeirotis)"),
    (re.compile(r"\btriantafyllopoulos\b", re.I), "author surname (Triantafyllopoulos)"),
    (re.compile(r"\bnyu\b", re.I), "institution (NYU)"),
    (re.compile(r"\bstern\.nyu\.edu\b", re.I), "institutional domain"),
    (re.compile(r"\baletheia\b", re.I), "web demo brand name"),
    (re.compile(r"\bnew york university\b", re.I), "institution name"),
    (re.compile(r"\bstern school\b", re.I), "institution name"),
    (re.compile(r"gs://autoencoders-census", re.I), "internal GCP bucket"),
]


REQUIRED_HARNESS_FILES = [
    "code/evaluate/detection.py",
    "code/evaluate/baselines.py",
    "code/experiment_b/fairness.py",
    "code/revision_experiments/lib_r2.py",
    "code/scripts/misq_revision/run_all_snap_experiments.py",
    "code/scripts/misq_revision/run_coherence_sim.py",
]


def run_git(repo: Path, args: List[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def git_rev_parse(repo: Path, ref: str = "HEAD") -> Optional[str]:
    proc = run_git(repo, ["rev-parse", "--verify", ref], check=False)
    if proc.returncode != 0:
        return None
    return proc.stdout.decode("utf-8").strip()


def resolve_experiment_ref(code_repo: Path, requested_ref: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Find a git ref containing experiment_b/ and revision_experiments/."""
    candidates: List[str] = []
    if requested_ref:
        candidates.append(requested_ref)
    candidates.extend([
        "HEAD",
        "revision/experiment-scripts",
        "origin/revision/experiment-scripts",
        "FETCH_HEAD",
    ])
    for ref in candidates:
        sha = git_rev_parse(code_repo, ref)
        if not sha:
            continue
        chk = run_git(code_repo, ["cat-file", "-e", f"{sha}:evaluate/detection.py"], check=False)
        chk2 = run_git(code_repo, ["cat-file", "-e", f"{sha}:revision_experiments/lib_r2.py"], check=False)
        if chk.returncode == 0 and chk2.returncode == 0:
            return ref, sha
    return None, None


def extract_git_path(repo: Path, ref_sha: str, rel_path: str, dest_root: Path) -> List[Path]:
    """Extract a file or directory from a git commit into dest_root/rel_path."""
    proc = run_git(repo, ["ls-tree", "-r", "--name-only", ref_sha, rel_path], check=False)
    if proc.returncode != 0:
        return []
    extracted: List[Path] = []
    for line in proc.stdout.decode("utf-8").splitlines():
        item = line.strip()
        if not item:
            continue
        p = Path(item)
        if any(part in SKIP_DIR_NAMES for part in p.parts) or p.suffix in SKIP_EXTENSIONS:
            continue
        blob = run_git(repo, ["show", f"{ref_sha}:{item}"]).stdout
        out_path = dest_root / item
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(blob)
        extracted.append(out_path)
    return extracted


def copy_tree_clean(src: Path, dst: Path) -> None:
    """Copy a directory tree while omitting caches and temporary artifacts."""
    for root, dirs, files in os.walk(src):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIR_NAMES)
        rel_root = Path(root).relative_to(src)
        target_dir = dst / rel_root
        target_dir.mkdir(parents=True, exist_ok=True)
        for fname in sorted(files):
            if fname in SKIP_DIR_NAMES or Path(fname).suffix in SKIP_EXTENSIONS:
                continue
            shutil.copy2(Path(root) / fname, target_dir / fname)


def sanitize_code_paths(code_root: Path, blind: bool) -> List[str]:
    """Replace hardcoded developer paths with environment-variable defaults."""
    notes: List[str] = []

    p1 = code_root / "experiment_b" / "pl_tradeoff_figs.py"
    if p1.exists():
        txt = p1.read_text(encoding="utf-8")
        old = 'FIG = "/Users/iliastriantafyllopoulos/Documents/my_projects/inattentiveness_paper/figures"'
        new = 'FIG = os.environ.get("FIG_DIR", os.path.join(os.getcwd(), "outputs", "figures"))'
        if old in txt:
            txt = txt.replace(old, new)
            if "os.makedirs(FIG, exist_ok=True)" not in txt:
                txt = txt.replace(new, new + '\nos.makedirs(FIG, exist_ok=True)')
            p1.write_text(txt, encoding="utf-8")
            notes.append("Sanitized hardcoded figure output path in code/experiment_b/pl_tradeoff_figs.py")

    p2 = code_root / "revision_experiments" / "fig_mm.py"
    if p2.exists():
        txt = p2.read_text(encoding="utf-8")
        old = 'OUT="/Users/iliastriantafyllopoulos/Documents/my_projects/inattentiveness_paper/figures"'
        new = 'OUT = os.environ.get("FIG_DIR", os.path.join(os.getcwd(), "outputs", "figures"))\nos.makedirs(OUT, exist_ok=True)'
        if old in txt:
            txt = txt.replace(old, new)
            p2.write_text(txt, encoding="utf-8")
            notes.append("Sanitized hardcoded figure output path in code/revision_experiments/fig_mm.py")

    p3 = code_root / "experiment_b" / "generate_detailed_tables_draft.py"
    if p3.exists():
        txt = p3.read_text(encoding="utf-8")
        old = '    out = "../inattentiveness_paper/sections/_regenerated_detailed_tables_draft.tex"'
        new = (
            '    out = os.environ.get("OUT_TEX", os.path.join(os.getcwd(), "outputs", "_regenerated_detailed_tables_draft.tex"))\n'
            '    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)'
        )
        if old in txt:
            txt = txt.replace(old, new)
            p3.write_text(txt, encoding="utf-8")
            notes.append("Parameterized LaTeX output path in code/experiment_b/generate_detailed_tables_draft.py")

    p_sim = code_root / "scripts" / "misq_revision" / "run_coherence_sim.py"
    if p_sim.exists():
        txt = p_sim.read_text(encoding="utf-8")
        old = '    fig.savefig(OUT_FIG, bbox_inches="tight")'
        new = (
            '    os.makedirs(os.path.dirname(os.path.abspath(OUT_FIG)), exist_ok=True)\n'
            '    fig.savefig(OUT_FIG, bbox_inches="tight")'
        )
        if old in txt and "os.makedirs(os.path.dirname(os.path.abspath(OUT_FIG))" not in txt:
            txt = txt.replace(old, new)
            p_sim.write_text(txt, encoding="utf-8")
            notes.append("Added parent directory creation before savefig in code/scripts/misq_revision/run_coherence_sim.py")

    p_snap_fig = code_root / "scripts" / "misq_revision" / "generate_snap_figures.py"
    if p_snap_fig.exists():
        txt = p_snap_fig.read_text(encoding="utf-8")
        old = 'FIG_DIR = os.environ.get("FIG_DIR", os.path.join(PAPER_DIR, "figures"))'
        new = old + '\nos.makedirs(FIG_DIR, exist_ok=True)'
        if old in txt and "os.makedirs(FIG_DIR, exist_ok=True)" not in txt:
            txt = txt.replace(old, new, 1)
            p_snap_fig.write_text(txt, encoding="utf-8")
            notes.append("Added FIG_DIR directory creation in code/scripts/misq_revision/generate_snap_figures.py")

    for sh_path in sorted((code_root / "experiment_b").glob("run_*.sh")) if (code_root / "experiment_b").is_dir() else []:
        txt = sh_path.read_text(encoding="utf-8")
        if "source venv_ae311/bin/activate" in txt:
            txt = txt.replace(
                "source venv_ae311/bin/activate",
                "if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi",
            )
            sh_path.write_text(txt, encoding="utf-8")
            notes.append(f"Normalized venv activation in code/experiment_b/{sh_path.name}")

    p4 = code_root / "scripts" / "misq_revision" / "README.md"
    if p4.exists() and blind:
        txt = p4.read_text(encoding="utf-8")
        txt = re.sub(
            r"\(companion PR \[`ipeirotis-org/inattentiveness_paper#32`\]\(https://github\.com/ipeirotis-org/inattentiveness_paper/pull/32\)\)",
            "(companion manuscript revision)",
            txt,
        )
        txt = txt.replace("`gs://autoencoders-census-paper-data/snapshots/2026-10-23`", "`<paper-data-snapshot>`")
        p4.write_text(txt, encoding="utf-8")
        notes.append("Anonymized repository/bucket references in code/scripts/misq_revision/README.md")

    return notes


def sanitize_verification_for_blind(verif_dir: Path) -> None:
    """Redact internal GCS bucket names and developer-path notes in blind mode."""
    for fname in ("REPORT.md", "verification-report.json"):
        fp = verif_dir / fname
        if not fp.exists():
            continue
        txt = fp.read_text(encoding="utf-8")
        txt = txt.replace("gs://autoencoders-census-paper-data/snapshots/2026-10-23", "<archived-data-snapshot>")
        fp.write_text(txt, encoding="utf-8")


def write_requirements_replication(code_root: Path) -> None:
    """Write a minimal, pinned scientific-Python requirements file (no GCP bloat)."""
    content = textwrap.dedent("""\
        # Minimal pinned Python environment for reproducing all paper analyses,
        # tables, figures, and simulations (verified on Python 3.11.2, Linux x86_64).
        # Excludes Google Cloud / web-app dependencies from requirements-full.txt.
        #
        # On Apple Silicon (arm64 macOS), replace tensorflow-cpu==2.15.1 with
        # tensorflow-macos==2.15.1 (or use Linux x86_64 / Docker).

        numpy==1.26.4
        pandas==2.2.3
        scipy==1.15.3
        scikit-learn==1.5.2
        tensorflow-cpu==2.15.1
        keras==2.15.0
        keras-tuner==1.4.8
        matplotlib==3.9.4
        seaborn==0.13.2
        PyYAML==6.0.3
        click==8.1.8
        joblib==1.5.3
        tabulate==0.9.0
        chardet==5.2.0
        pymupdf==1.24.14
        pytest>=8.0.0
    """)
    (code_root / "requirements-replication.txt").write_text(content, encoding="utf-8")


def hydrate_from_snapshot(
    snapshot_dir: Path,
    code_root: Path,
    include_third_party_raw: bool,
    include_weights: bool,
) -> Dict[str, object]:
    """Hydrate precomputed score caches, derived CSVs (and optionally raw CSVs) from a snapshot."""
    summary: Dict[str, object] = {
        "hydrated": True,
        "snapshot_dir": str(snapshot_dir),
        "cache_files_copied": 0,
        "derived_csvs_copied": [],
        "raw_csvs_copied": [],
        "weights_included": include_weights,
    }
    src_cache = snapshot_dir / "cache"
    dst_cache = code_root / "cache"
    if src_cache.is_dir():
        count = 0
        for root, dirs, files in os.walk(src_cache):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIR_NAMES)
            rel_root = Path(root).relative_to(src_cache)
            # Skip heavy TF SavedModel weight folders unless explicitly requested
            if not include_weights and ("autoencoder" in rel_root.parts or "variables" in rel_root.parts):
                continue
            for fname in sorted(files):
                if not include_weights and fname not in SCORE_CACHE_ALLOW_FILENAMES:
                    continue
                target = dst_cache / rel_root / fname
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(Path(root) / fname, target)
                count += 1
        summary["cache_files_copied"] = count

    copied_derived: List[str] = []
    for derived_name in DERIVED_SNAPSHOT_CSVS:
        cand = snapshot_dir / derived_name
        if cand.is_file():
            shutil.copy2(cand, code_root / derived_name)
            copied_derived.append(derived_name)
    summary["derived_csvs_copied"] = copied_derived

    if include_third_party_raw:
        src_data = snapshot_dir / "data"
        dst_data = code_root / "data"
        copied_csvs: List[str] = []
        if src_data.is_dir():
            for csv_name in EXTERNAL_BENCHMARK_CSVS:
                cand = src_data / csv_name
                if cand.is_file():
                    dst_data.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(cand, dst_data / csv_name)
                    copied_csvs.append(csv_name)
        summary["raw_csvs_copied"] = copied_csvs

    return summary


def build_readme(mode: str, version: str, hydration: Dict[str, object], pkg_name: str) -> str:
    blind = mode == "blind"
    authors_line = (
        "Authors withheld for double-anonymous review"
        if blind
        else "Ilias Triantafyllopoulos and Panos Ipeirotis (Stern School of Business, New York University)"
    )
    demo_section = (
        ""
        if blind
        else textwrap.dedent("""\
            ## Companion Software and Web Application

            - **Live web application (Aletheia):** <https://aletheia.stern.nyu.edu/> (CSV upload for label-free survey screening).
            - **Active source repositories:**
              - Code and web platform: <https://github.com/ipeirotis-org/autoencoders_census>
              - Manuscript and Lean 4 formalization: <https://github.com/ipeirotis-org/inattentiveness_paper>

        """)
    )

    derived_csvs = set(hydration.get("derived_csvs_copied") or [])
    has_sample_size_csv = "sample_size_results.csv" in derived_csvs

    if hydration.get("hydrated"):
        derived_note = (
            f" and {len(derived_csvs)} derived experiment summary CSV(s) "
            f"({', '.join(f'`code/{n}`' for n in sorted(derived_csvs))})"
            if derived_csvs
            else ""
        )
        cache_note = (
            f"This archive **includes** {hydration['cache_files_copied']} hydrated score-cache and "
            f"hyperparameter-tuning files under `code/cache/`{derived_note} for exact score-level reproduction."
        )
    else:
        cache_note = (
            "This archive ships the complete code, verified aggregate result JSONs (`results/`), "
            "the 242-entry number-verification ledger (`results/verification/`), the zero-dependency "
            "Lean 4 formalization (`lean/`), and the public-domain 2017 CDC YRBS survey "
            "(`code/data/sadc_2017only_national_full.csv`). Per MIS Quarterly Policy §4.5 (Data Rights "
            "and Provenance), raw CSVs for the eight third-party academic surveys are **not** "
            "redistributed directly without confirming per-source licenses; see `DATA_PROVENANCE.md` "
            "for canonical DOIs/URLs and exact placement paths under `code/data/`. Re-running "
            "`replication/build_zenodo_package.py --snapshot-dir <path>` hydrates the full "
            "per-respondent score caches (`code/cache/`) and derived experiment CSVs "
            "(`code/sample_size_results.csv`, etc.) for exact score-level replay."
        )

    if hydration.get("hydrated") and has_sample_size_csv:
        step5_cache_note = (
            "This archive already ships hydrated `code/cache/` score caches, tuned configs, and "
            "`code/sample_size_results.csv`, so Steps 5a–5b are optional (only needed if retraining "
            "models or regenerating learning curves from scratch)."
        )
        step5_commands = textwrap.dedent("""\
            # Run from code/ with .venv active
            # Steps 5a-5b (optional when code/cache/ and sample_size_results.csv are pre-hydrated): refit from scratch
            for ds in attention_check bot_bot_mturk inattentive moral_data mturk_ethics pennycook_1 public_opinion racial_data; do
              python main.py chow_liu_outliers --data "$ds" --output "cache/_fix9_${ds}_cl/"
            done
            bash experiment_b/run_tuning.sh
            bash experiment_b/run_finalize.sh
            bash experiment_b/run_multiseed.sh
            python -m experiment_b.linear_ae
            python -m experiment_b.sample_size""")
    elif hydration.get("hydrated"):
        step5_cache_note = (
            "**Prerequisite (`code/sample_size_results.csv`):** This archive ships hydrated `code/cache/` "
            "score caches and tuned configs (so Step 5a model refits are optional), **but** "
            "`code/sample_size_results.csv` was not present in the snapshot. Both `run_all_snap_experiments.py` "
            "and `generate_snap_figures.py` read `sample_size_results.csv`, so you **must** run Step 5b "
            "(`python -m experiment_b.sample_size`) before executing Steps 5c–5d."
        )
        step5_commands = textwrap.dedent("""\
            # Run from code/ with .venv active
            # Step 5a (optional when code/cache/ is pre-hydrated): refit detector caches from scratch
            for ds in attention_check bot_bot_mturk inattentive moral_data mturk_ethics pennycook_1 public_opinion racial_data; do
              python main.py chow_liu_outliers --data "$ds" --output "cache/_fix9_${ds}_cl/"
            done
            bash experiment_b/run_tuning.sh
            bash experiment_b/run_finalize.sh
            bash experiment_b/run_multiseed.sh
            python -m experiment_b.linear_ae

            # Step 5b (REQUIRED when sample_size_results.csv is not bundled): generate learning-curve CSV
            python -m experiment_b.sample_size""")
    else:
        step5_cache_note = (
            "**Prerequisites (`code/cache/` & `code/sample_size_results.csv`):** `run_all_snap_experiments.py` and "
            "`generate_snap_figures.py` read tuned hyperparameters (`cache/_tuned_<ds>_ae/best_hyperparameters.yaml`), "
            "detector score caches (`cache/_fix9_<ds>_cl/errors.csv`, `cache/_tuned_<ds>_ae_p{85,100}/errors.csv`, "
            "`cache/_ms_<ds>_p{85,100}_s{1..5}/errors.csv`, `cache/_lin0_<ds>/errors.csv`), and the sample-size "
            "learning-curve table (`sample_size_results.csv`). Because this archive was built without "
            "`--snapshot-dir`, you **must** run Steps 5a–5b first (or rebuild with `--snapshot-dir` against the "
            "snapshot) before executing Steps 5c–5d."
        )
        step5_commands = textwrap.dedent("""\
            # Run from code/ with .venv active
            # Steps 5a-5b (REQUIRED when code/cache/ and sample_size_results.csv are not pre-hydrated): populate caches & learning-curve CSV
            for ds in attention_check bot_bot_mturk inattentive moral_data mturk_ethics pennycook_1 public_opinion racial_data; do
              python main.py chow_liu_outliers --data "$ds" --output "cache/_fix9_${ds}_cl/"
            done
            bash experiment_b/run_tuning.sh
            bash experiment_b/run_finalize.sh
            bash experiment_b/run_multiseed.sh
            python -m experiment_b.linear_ae
            python -m experiment_b.sample_size""")

    step5_block = (
        f"{step5_cache_note}\n\n"
        "```bash\n"
        f"{step5_commands}\n\n"
        "# Step 5c: Run unified 9-survey ablations, classical one-hot baselines,\n"
        "# fairness audits, sample-size curves, and downstream analyses\n"
        "SNAP_REPO=$PWD python scripts/misq_revision/run_all_snap_experiments.py\n\n"
        "# Step 5d: Regenerate figures into ../outputs/figures/\n"
        "mkdir -p ../outputs/figures\n"
        "SNAP_REPO=$PWD FIG_DIR=../outputs/figures python scripts/misq_revision/generate_snap_figures.py\n"
        "```"
    )

    body = textwrap.dedent(f"""\
        # Replication Package: *{PAPER_TITLE}*

        **Version:** `{version}` ({mode})
        **Authors:** {authors_line}
        **Target Venue:** *MIS Quarterly*

        > **MIS Quarterly Research Transparency Disclaimer:** These transparency and replication
        > materials are made available for the benefit of readers to support verification,
        > reproducibility, and reuse. They were not subject to peer review by *MIS Quarterly*, and
        > *MIS Quarterly* bears no responsibility for their use. Every primary claim, empirical
        > table, figure, and analytical proof (Propositions 1–2, Appendices A–B) is contained
        > within the peer-reviewed manuscript itself.

        ## Package Overview

        {cache_note}

        ```text
        {pkg_name}/
        ├── README.md                    # This file (setup, quick start, disclaimers)
        ├── REPRODUCTION_MAP.md          # Complete map: manuscript tables, figures & theorems -> code/Lean
        ├── DATA_PROVENANCE.md           # Sources, DOIs, licenses, and placement guide for all 9 surveys
        ├── LICENSE                      # MIT License (code and Lean 4 proofs)
        ├── LICENSE-DOCS.txt             # CC-BY-4.0 notice (documentation, supplement, verified outputs)
        ├── MANIFEST.json                # Machine-readable provenance and build metadata
        ├── SHA256SUMS.txt               # SHA-256 cryptographic checksums for all files
        ├── code/                        # Unified Python package + experiment & figure scripts
        │   ├── main.py                  # CLI entry point (AE, Linear AE, Chow-Liu)
        │   ├── chow_liu_rank.py         # Closed-form Chow-Liu tree scorer (Laplace alpha=1)
        │   ├── dataset/                 # DataLoader (Rule-of-N encoding, >=20 numeric binning)
        │   ├── features/                # Table2Vector one-hot vectorizer
        │   ├── model/                   # AutoencoderModel, Percentile Loss, LinearAutoencoder
        │   ├── evaluate/                # detection.py (aligned labels/metrics), baselines.py (6 indices)
        │   ├── config/                  # YAML hyperparameter configs & search spaces
        │   ├── experiment_b/            # Tuning, linear/PL-AE finalization, fairness & PL-sweep scripts
        │   ├── revision_experiments/    # Measurement-model verification & downstream sharpening scripts
        │   ├── scripts/misq_revision/   # Unified MISQ revision scripts (coherence sim, YRBS, 9-survey)
        │   ├── tests/                   # Unit & integration test suite (pytest)
        │   ├── data/                    # Public-domain CDC 2017 YRBS CSV (plus external CSV placement dir)
        │   └── requirements-replication.txt
        ├── lean/                        # Zero-dependency Lean 4 formalization (v4.35.0, 0 sorry, 0 axiom)
        ├── results/                     # Verified revision JSON summaries & 242-entry audit ledger
        │   ├── coherence_sim_results.json
        │   ├── yrbs_revision_results.json
        │   ├── snap_revision_results.json
        │   └── verification/            # REPORT.md, results.csv, verification-report.json
        └── docs/
            └── MISQ_online_supplement.pdf  # Unreviewed technical reference documentation (47 pages)
        ```

        ## Quick Start

        Start from the unpacked package root directory (`{pkg_name}/`).

        ### 1. Verify Package Integrity (SHA-256)

        ```bash
        sha256sum -c SHA256SUMS.txt
        ```

        ### 2. Verify Machine-Checked Proofs in Lean 4 (~5 seconds, zero external dependencies)

        Requires Lean 4 `v4.35.0` (exact toolchain pinned in `lean/lean-toolchain`, e.g., via `elan`).
        No Mathlib or external package downloads are required. Run in a subshell to remain at the
        package root:

        ```bash
        (cd lean && lake build)
        ```

        All five modules under `lean/InattentivenessProofs/` (`OrderedField.lean`, `Basic.lean`,
        `MeasurementModel.lean`, `Fairness.lean`, `CostBenefit.lean`) compile with **zero `sorry`
        placeholders** and **zero custom `axiom` declarations**. Scope note: `ProbSpace` formalizes
        arbitrary finite-support probability distributions (including any finite population or
        empirical survey law) over an abstract `OrderedField F`; continuous-population and general
        Borel-measurable extensions in Propositions 1–2 are proved analytically in Appendices A–B of
        the manuscript.

        ### 3. Set Up Python Environment (Python 3.11 recommended)

        Enter `code/` once and activate `.venv` (Steps 4–6 run from `code/` in the same shell):

        ```bash
        cd code
        python3.11 -m venv .venv
        source .venv/bin/activate
        pip install --upgrade pip
        pip install -r requirements-replication.txt
        ```

        *(For exact neural-network reproducibility notes, see `docs/MISQ_online_supplement.pdf`,
        Section S18: Chow-Liu is closed-form and reproduces bit-identically; TensorFlow neural
        retraining reproduces up to small training-seed variance bounded by the reported bootstrap
        and 5-seed intervals.)*

        ### 4. Run Standalone Experiments Immediately (No External Downloads Required)

        Both the controlled 4-panel coherence simulation (Section 7.5, Figure 5) and the full
        92-variable composite-disjoint YRBS analysis (Kann et al., 2018; Tables 2–4 and Sections
        7.1, 7.8, 7.9) run directly out of the box using the bundled public-domain CDC 2017 YRBS
        dataset:

        ```bash
        # Run from code/ with .venv active
        # Unit & integration test suite
        PYTHONPATH=$PWD pytest tests/ -v

        # Reproduce Section 7.5 / Figure 5 controlled coherence simulation
        mkdir -p ../outputs/figures
        FIG_DIR=../outputs/figures OUT_FIG=../outputs/figures/coherence_simulation.pdf \\
          python scripts/misq_revision/run_coherence_sim.py

        # Reproduce Kann et al. (2018) 92-variable disjoint YRBS analyses
        python scripts/misq_revision/run_yrbs_revision.py
        ```

        ### 5. Run Full 9-Survey Benchmark Suite

        After placing the eight third-party survey CSVs under `code/data/` (see `DATA_PROVENANCE.md`
        for canonical repositories and exact filenames):

        __STEP5_BLOCK__

        ### 6. Screen Your Own Survey CSV (Label-Free)

        ```bash
        # Run from code/ with .venv active
        python main.py chow_liu_outliers --data /path/to/your_survey.csv --output ./screening_out/
        ```

        __DEMO_SECTION__## Licensing

        - **Software & Lean 4 Proofs (`code/`, `lean/`):** MIT License (`LICENSE`).
        - **Documentation, Verified Result Summaries & Technical Supplement (`README.md`, `*.md`, `results/`, `docs/`):** Creative Commons Attribution 4.0 International (`CC-BY-4.0`, `LICENSE-DOCS.txt`).
        - **Bundled CDC 2017 YRBS Dataset (`code/data/sadc_2017only_national_full.csv`):** U.S. Federal Government Public Domain (17 U.S.C. § 105).
        - **External Benchmark Datasets:** Governed by each original study's repository license (`DATA_PROVENANCE.md`).
    """)
    return body.replace("__STEP5_BLOCK__", step5_block).replace("__DEMO_SECTION__", demo_section)


def build_reproduction_map() -> str:
    return textwrap.dedent("""\
        # Claim, Table, Figure, and Theorem Reproduction Map

        This index maps every numbered table, figure, proposition, and analytical claim in the
        *MIS Quarterly* manuscript (`MISQ_submission_55.pdf`) and the unreviewed technical supplement
        (`docs/MISQ_online_supplement.pdf`) to the exact code script, Lean 4 declaration, and
        verified numerical output in this archive.

        ## 1. Main Manuscript Tables

        | Manuscript Item | Description | Primary Script(s) | Verified Output(s) |
        |---|---|---|---|
        | **Table 1** (Sec. 3, p. 10) | Benchmark survey characteristics (`Samples`, `Variables`, `Features`, `AFV`) across 9 datasets | `code/dataset/loader.py`, `code/features/transform.py`, `code/afv_lift.py` | `results/verification/REPORT.md` (`D01`–`D40`, `H4`) |
        | **Table 2** (Sec. 6.3, p. 28) | Label-free detection AUC across 9 surveys: Chow-Liu, PL-AE (`p=85`), Base AE (`p=100`), Linear AE (`p=100`, `p=85`), and 6 psychometric indices | `code/experiment_b/final_table.py`, `code/scripts/misq_revision/run_all_snap_experiments.py`, `code/scripts/misq_revision/run_yrbs_revision.py` | `results/snap_revision_results.json`, `results/yrbs_revision_results.json`, `results/verification/results.csv` |
        | **Table 3** (Sec. 7.1, p. 31) | Empirical verification of the measurement model (observed correlations, Corollary 1 lower bounds, multi-check triad factor loadings, single-factor RMSR) | `code/revision_experiments/theory_verify.py`, `code/revision_experiments/theory_all.py`, `code/scripts/misq_revision/run_all_snap_experiments.py` | `results/snap_revision_results.json`, `results/yrbs_revision_results.json`, `results/verification/results.csv` |
        | **Table 4** (Sec. 7.9, p. 42) | Downstream substantive effects across 6 replicated studies (in-sample & out-of-outcome trimming vs. check exclusion) | `code/revision_experiments/expE_strength.py`, `code/revision_experiments/expE_robinson.py`, `code/scripts/misq_revision/run_all_snap_experiments.py`, `code/scripts/misq_revision/run_yrbs_revision.py` | `results/snap_revision_results.json`, `results/yrbs_revision_results.json` |

        ## 2. Main Manuscript Figures

        | Figure | Description | Generator / Source |
        |---|---|---|
        | **Figure 1** (Sec. 1, p. 2) | End-to-end unsupervised screening workflow diagram | Conceptual architecture diagram (LaTeX/draw.io in manuscript) |
        | **Figure 2** (Sec. 4.1, p. 13) | Measurement-model causal DAG ($A, C, M, U_C, U_M$) | TikZ diagram in manuscript (`sections/theory.tex`) |
        | **Figure 3** (Sec. 4.2, p. 15) | Augmented fairness-audit DAG ($Z, A, C, M$) | TikZ diagram in manuscript (`sections/theory.tex`) |
        | **Figure 4** (Sec. 6, p. 25) | Reconstruction ORA comparison across models & surveys | `code/experiment_b/reconstruction_table.py`, `code/scripts/misq_revision/update_figures.py` |
        | **Figure 5** (Sec. 7.5, p. 35) | 4-panel controlled coherence simulation (loading $\\lambda$, breadth $J$, random carelessness, straightlining) | `code/scripts/misq_revision/run_coherence_sim.py` -> `results/coherence_sim_results.json` |
        | **Figure 6** (Sec. 7.6, p. 36) | Percentile Loss reconstruction-vs-detection trade-off (Lift & AUC boxplots) | `code/experiment_b/pl_sweep.py`, `code/experiment_b/pl_tradeoff_figs.py` |
        | **Figure 7** (Sec. 7.7, p. 37) | Fixed-$N \\ge 1{,}000$ cohort learning curves across sample sizes | `code/experiment_b/sample_size.py`, `code/scripts/misq_revision/generate_snap_figures.py` |
        | **Figure 8** (Sec. 7.8, p. 39) | Detection-vs-fairness frontier across detectors and psychometric baselines | `code/experiment_b/fairness.py`, `code/experiment_b/fairness_baselines.py`, `code/scripts/misq_revision/update_figures.py` |
        | **Figure 9** (Sec. 7.9, p. 42) | Downstream effect sharpening curves (in-sample & out-of-outcome) | `code/revision_experiments/expE_strength.py`, `code/scripts/misq_revision/generate_snap_figures.py` |

        ## 3. Theoretical Results and Machine-Checked Lean 4 Proofs

        Full human-readable analytical proofs of Propositions 1 and 2 (including continuous-population
        and general Borel-measurable cases) appear in Appendices A and B of the main manuscript.
        All algebraic identities, ordered-field bounds, finite-support probability propositions,
        sharpness counterexamples, and cost-benefit dominance rules are independently machine-checked
        under `lean/InattentivenessProofs/` (`lake build`, Lean 4 `v4.35.0`, 0 `sorry`, 0 custom `axiom`):

        | Manuscript Result | Lean 4 Module | Main Declaration(s) |
        |---|---|---|
        | Ordered-field typeclass & constructive $\\mathbb{Q}$ witness | `OrderedField.lean` | `OrderedField`, `instOrderedFieldRat` |
        | Finite-support probability spaces, `CondIndep`, tie-aware ROC / Somers' kernel | `Basic.lean` | `ProbSpace`, `condIndep_iff_condProb`, `expect_prod_self_antisymm`, `two_mul_auc_sub_one` |
        | **Proposition 1** (Sec. 4.1, App. A): automatic linearity of $E[X\\mid A]$ for binary $A$ | `MeasurementModel.lean` | `binary_cond_expect_linear` |
        | **Proposition 1** (Eq. 1): linear correlation decomposition | `MeasurementModel.lean` | `corr_proxy_decomp`, `corr_sq_proxy_decomp` |
        | **Proposition 1**: binary rank / AUC decomposition with score ties | `MeasurementModel.lean` | `auc_proxy_decomp`, `auc_proxy_decomp_of_condIndep` |
        | **Corollary 1** (Sec. 4.1): attenuation lower bounds on latent $\\mathrm{Corr}$ and $\\mathrm{AUC}$ | `MeasurementModel.lean` | `corr_latent_lower_bound`, `auc_latent_lower_bound` |
        | **Section 7.1**: multi-check $(M, C_i, C_j)$ triad loading identity | `MeasurementModel.lean` | `triad_squared_loading`, `triad_loading_eq` |
        | **Figure 3**: Markov contractions ((F1)+(F2) $\\Rightarrow$ (A1); (F1)–(F3) $\\Rightarrow$ $M \\perp\\!\\!\\!\\perp Z \\mid (A,C)$) | `Fairness.lean` | `condIndep_CM_given_A_of_F1_F2`, `condIndep_MZ_given_AC_of_F1_F2_F3` |
        | **Proposition 2 & Corollary 2** (Sec. 4.2, App. B): proxy fairness & falsification | `Fairness.lean` | `latent_fair_implies_proxy_fair`, `proxy_unfair_falsifies_conjunction` |
        | Sharpness of (F1): 8-state XOR counterexample under $C \\perp\\!\\!\\!\\perp M \\mid A$ alone | `Fairness.lean` | `xor_counterexample`, `xor_counterexample_rat` |
        | **Appendix B Remark**: linear-Gaussian partial-correlation bound & sharpness of $\\mathrm{Var}(M\\mid A) > 0$ | `Fairness.lean` | `partial_corr_MZ_sq_bound`, `partial_corr_MZ_vanishes_eps_delta`, `partial_corr_MA_sq_degenerate_one` |
        | **Section 8.1** (Eq. 4) & **Supplement S13** (Eq. S5): screening & hybrid dominance rules | `CostBenefit.lean` | `unsupervised_dominates_checks_iff`, `hybrid_dominates_pure_arm_iff`, `hybrid_dominates_both_of_stronger` |

        ## 4. Technical Supplement Tables (`docs/MISQ_online_supplement.pdf`)

        | Supplement Section / Table | Content | Primary Script(s) |
        |---|---|---|
        | **S3 (Table S1, Fig. S2)** | Dataset-level correlations & predictor scatter plots | `code/experiment_b/recompute_stats.py`, `code/make_plots.py` |
        | **S4 (Fig. S3)** | Full 9-metric Percentile Loss trade-off panels | `code/experiment_b/pl_sweep.py`, `code/experiment_b/pl_tradeoff_figs.py` |
        | **S5 (Tables S2–S3)** | Variable coverage, one-hot classical MD/PCA/MCA, and codebook-free PC1 sign alignment | `code/scripts/misq_revision/run_all_snap_experiments.py`, `code/scripts/misq_revision/run_yrbs_revision.py` |
        | **S6 (Table S4)** | Same-input robustness (our models restricted to the ordinal Likert battery) | `code/experiment_b/same_input.py` |
        | **S7, S14–S16 (Tables S5, S13–S15)** | Full per-population fairness, comparative disparity, opinion minorities, high-variance responders, aggregate fairness | `code/experiment_b/fairness.py`, `code/experiment_b/fairness_baselines.py` |
        | **S9–S10 (Tables S6–S9)** | Full reconstruction & per-condition detection metrics | `code/experiment_b/reconstruction_table.py`, `code/experiment_b/build_detailed_tables.py` |
        | **S12 (Tables S10–S12)** | Encoding rules, hyperparameter search space, and KerasTuner selected architectures | `code/config/hp_autoencoder_real.yaml`, `code/scripts/misq_revision/run_all_snap_experiments.py` |
        | **S17 (Tables S16–S17)** | Bootstrap confidence intervals & 5-fold inductive (out-of-sample) Chow-Liu validation | `code/revision_experiments/favorable.py`, `code/scripts/misq_revision/run_yrbs_revision.py` |
        | **S19–S20 (Tables S18–S19)** | Missingness sensitivity & score-correlation / top-$h$ Jaccard overlap | `code/revision_experiments/favorable.py`, `code/scripts/misq_revision/run_all_snap_experiments.py` |
        | **S21 (Table S20)** | Complete Lean 4 theorem correspondence table | `lean/README.md`, `lean/InattentivenessProofs/*.lean` |
    """)


def build_data_provenance(hydration: Dict[str, object]) -> str:
    copied_raw = set(hydration.get("raw_csvs_copied") or [])
    raw_status = lambda fn: "Included in `code/data/`" if fn in copied_raw else "Obtain from original repository (place in `code/data/`)"

    return textwrap.dedent(f"""\
        # Benchmark Dataset Provenance and Licensing

        All empirical analyses evaluate nine publicly documented survey datasets that retain both
        attentive and inattentive respondents (uncleaned releases). In compliance with *MIS
        Quarterly* Publication Policy §4.5 (Data Rights and Provenance), this replication archive
        bundles the U.S. Federal Government public-domain CDC 2017 Youth Risk Behavior Survey
        directly and documents the canonical source publication, public data repository, expected
        filename, and access/licensing terms for the remaining eight third-party surveys.

        ## Benchmark Inventory (Table 1 of Manuscript)

        | # | Citation | Internal Key | Expected File under `code/data/` | $N$ (Modeled) | Vars / One-Hot | Original Paper DOI | Canonical Public Data Repository | Bundle Status & License Notes |
        |---|---|---|---|---:|---:|---|---|---|
        | 1 | Kann et al. (2018) — 2017 National YRBS | `sadc_2017` | `sadc_2017only_national_full.csv` | 14,765 | 98 (92 disjoint) / 619 | `10.15585/mmwr.ss6708a1` | CDC YRBSS Combined Datasets (<https://www.cdc.gov/yrbs/data/index.html>; 2017 Guide: <https://stacks.cdc.gov/view/cdc/55994>) | **Included in `code/data/`** — U.S. Federal Government Public Domain (17 U.S.C. § 105) |
        | 2 | Pennycook et al. (2020) — COVID-19 Misinformation (Study 1, Condition 1) | `pennycook_1` | `Pennycook et al._Study 1.csv` | 212 | 92 / 323 | `10.1177/0956797620939054` | OSF Project: <https://osf.io/7d3xh/> | {raw_status("Pennycook et al._Study 1.csv")} (check OSF node license terms before redistribution) |
        | 3 | Alvarez et al. (2019) — California Political Attitudes | `inattentive` | `inattentive_users.csv` | 2,725 | 37 / 173 | `10.1017/pan.2018.57` | Harvard Dataverse: <https://doi.org/10.7910/DVN/TUUYLQ> | {raw_status("inattentive_users.csv")} (export original replication data table to CSV; Dataverse terms apply) |
        | 4 | Uhalt (2020) — Attention Checks & Response Quality | `attention_check` | `attention_check.csv` | 308 | 60 / 337 | `10.17605/osf.io/uz872` | OSF Node: <https://osf.io/uz872/> *(note: node access status was restricted on 2026-11-09; see note below)* | {raw_status("attention_check.csv")} |
        | 5 | O'Grady et al. (2019) — Moral Foundations (Study 1) | `moral_data` | `moral_data.csv` | 355 | 72 / 325 | `10.1016/j.jrp.2019.06.008` | Mendeley Data: <https://data.mendeley.com/datasets/sbmwmsynxk> | {raw_status("moral_data.csv")} (Mendeley Data CC-BY terms apply) |
        | 6 | Buchanan & Scofield (2018) — Low-Quality Data / Bots | `bot_bot_mturk` | `Bot_Bot_Bot__MTURK.csv` | 1,038 | 20 / 135 | `10.3758/s13428-018-1035-6` | OSF Project: <https://osf.io/x6t8a/> (preprint: <https://osf.io/cv2bn/>) | {raw_status("Bot_Bot_Bot__MTURK.csv")} (OSF repository terms apply) |
        | 7 | Moss et al. (2023) — Ethical MTurk Survey (Study 1) | `mturk_ethics` | `ethics.csv` | 2,277 | 35 / 112 | `10.3758/s13428-022-02005-0` | OSF Project: <https://osf.io/8nhyz/> | {raw_status("ethics.csv")} (OSF repository terms apply) |
        | 8 | Mastroianni & Dana (2022) — Attitude Change (Study 1 completers) | `public_opinion` | `public_opinion.csv` | 1,003 (1,036 raw) | 49 / 222 | `10.1073/pnas.2107260119` | OSF Project: <https://osf.io/wud9a/> | {raw_status("public_opinion.csv")} (OSF repository terms apply; loader filters to completed surveys via `lookup` / `Finished` + `consent`) |
        | 9 | Ivanov et al. (2021) — Racial Resentment & Decarceration | `racial_data` | `racial_data.csv` | 860 | 67 / 310 | `10.1177/23780231211005222` | Published Open-Access *Socius* article & authors' replication materials | {raw_status("racial_data.csv")} (contact original authors / check *Socius* supplementary archive) |

        ## Notes on Dataset Handling and Reproducibility

        1. **Immediate standalone verification:** `sadc_2017only_national_full.csv` (CDC 2017 YRBS,
           $N = 14,765$) is included directly in `code/data/`, allowing immediate execution of
           `code/scripts/misq_revision/run_coherence_sim.py` (synthetic Monte Carlo) and
           `code/scripts/misq_revision/run_yrbs_revision.py` (full 92-variable composite-disjoint
           YRBS benchmark, fairness audit, covariate-adjusted validity, and drug-disparity
           downstream analysis).
        2. **Score caches vs. third-party raw CSVs:** Precomputed per-respondent detector scores
           (`errors.csv`) contain only anonymous row indices and scalar model atypicality scores (no
           raw survey text or PII). When this archive is built with `--snapshot-dir` on the authors'
           archived snapshot, `code/cache/` contains these derived score caches alongside the
           selected KerasTuner configurations (`best_hyperparameters.yaml`), permitting exact
           verification of every reported number even where third-party raw CSVs are fetched
           separately from their original repositories.
        3. **Uhalt (2020) OSF availability note:** The original OSF node (`https://osf.io/uz872/`)
           was publicly accessible at the time of initial data acquisition and became access-restricted
           on later verification (2026-11-09). Derived aggregate outputs (`results/snap_revision_results.json`
           and `results/verification/results.csv`) document all metrics on this benchmark, and all
           primary findings hold identically on the remaining eight openly accessible surveys.
        4. **Encoding conventions:** Every loader applies `dataset/loader.py`'s unified cleaning
           pipeline (retaining numeric columns with $<20$ distinct values as discrete categories,
           discretizing numeric columns with $\\ge 20$ distinct values into $z$-score bins at
           $\\pm 0.7, \\pm 1.4$ plus an explicit `"missing"` category, and dropping constant or
           high-cardinality columns under the Rule-of-$N$ filter). Attention-check columns are read
           raw and aligned via `code/evaluate/detection.py`.
    """)


def build_mit_license(mode: str) -> str:
    holder = "The Authors (anonymized for peer review)" if mode == "blind" else "Ilias Triantafyllopoulos and Panos Ipeirotis"
    year = date.today().year
    return textwrap.dedent(f"""\
        MIT License

        Copyright (c) {year} {holder}

        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.
    """)


def build_docs_license() -> str:
    return textwrap.dedent("""\
        Documentation, Technical Supplement, and Derived Result Summaries License

        Unless otherwise noted (such as the U.S. Federal Government public-domain CDC 2017 YRBS
        dataset under 17 U.S.C. § 105, and third-party datasets referenced in DATA_PROVENANCE.md),
        all documentation files (README.md, REPRODUCTION_MAP.md, DATA_PROVENANCE.md), verified
        aggregate result summaries (results/), and the unreviewed technical supplement PDF
        (docs/MISQ_online_supplement.pdf) in this archive are licensed under the Creative Commons
        Attribution 4.0 International License (CC BY 4.0).

        Full license text: https://creativecommons.org/licenses/by/4.0/legalcode
    """)


def build_citation_cff(version: str) -> str:
    today = date.today().isoformat()
    return textwrap.dedent(f"""\
        cff-version: 1.2.0
        message: "If you use this replication package, code, or Lean 4 formalization, please cite the accompanying manuscript and this archive."
        title: "Replication Package: {PAPER_TITLE}"
        type: software
        version: "{version}"
        date-released: "{today}"
        license: MIT
        repository-code: "https://github.com/ipeirotis-org/autoencoders_census"
        url: "https://aletheia.stern.nyu.edu/"
        authors:
          - family-names: "Triantafyllopoulos"
            given-names: "Ilias"
            affiliation: "Stern School of Business, New York University"
          - family-names: "Ipeirotis"
            given-names: "Panos"
            affiliation: "Stern School of Business, New York University"
        keywords:
          - "survey data quality"
          - "inattentive respondents"
          - "careless responding"
          - "autoencoders"
          - "percentile loss"
          - "Chow-Liu trees"
          - "algorithmic fairness"
          - "Lean 4"
          - "psychometrics"
    """)


def build_zenodo_json(version: str) -> dict:
    today = date.today().isoformat()
    return {
        "title": f"Replication Package: {PAPER_TITLE}",
        "upload_type": "software",
        "publication_date": today,
        "version": version,
        "description": (
            f"<p>Replication and research-transparency archive accompanying the manuscript "
            f"<em>{PAPER_TITLE}</em>.</p>"
            f"<p><strong>Contents:</strong> (1) Unified Python package and experiment scripts for "
            f"label-free survey inattentiveness detection using closed-form Chow-Liu dependence trees, "
            f"Percentile-Loss autoencoders, Linear autoencoders, six psychometric careless-responding "
            f"baselines, and the subgroup-fairness audit; (2) verified numerical result summaries and "
            f"a 242-entry manuscript number-verification audit ledger; (3) the U.S. public-domain "
            f"2017 CDC YRBS benchmark dataset and provenance/licensing documentation for all nine "
            f"benchmark surveys; (4) a zero-dependency Lean 4 (v4.35.0) formalization with zero "
            f"<code>sorry</code> placeholders machine-checking the measurement-model, fairness-audit, "
            f"and screening-economics identities; and (5) unreviewed technical reference documentation "
            f"(<code>MISQ_online_supplement.pdf</code>).</p>"
            f"<p><strong>Companion live web application:</strong> "
            f"<a href=\"https://aletheia.stern.nyu.edu/\">https://aletheia.stern.nyu.edu/</a></p>"
            f"<p><em>MIS Quarterly Research Transparency Notice:</em> These materials are provided "
            f"for the benefit of readers, were not subject to peer review by MIS Quarterly, and "
            f"MIS Quarterly bears no responsibility for their use.</p>"
        ),
        "creators": [
            {
                "name": "Triantafyllopoulos, Ilias",
                "affiliation": "Stern School of Business, New York University",
            },
            {
                "name": "Ipeirotis, Panos",
                "affiliation": "Stern School of Business, New York University",
            },
        ],
        "access_right": "open",
        "license": "MIT",
        "keywords": [
            "survey data quality",
            "inattentive respondents",
            "careless responding",
            "autoencoders",
            "percentile loss",
            "Chow-Liu trees",
            "measurement model",
            "algorithmic fairness",
            "Lean 4",
            "reproducibility",
        ],
        "related_identifiers": [
            {
                "identifier": "https://github.com/ipeirotis-org/autoencoders_census",
                "relation": "isSupplementTo",
                "resource_type": "software",
            },
            {
                "identifier": "https://aletheia.stern.nyu.edu/",
                "relation": "isDocumentedBy",
                "resource_type": "software",
            },
            {
                "identifier": "https://arxiv.org/abs/2603.02427",
                "relation": "isSupplementTo",
                "resource_type": "publication-preprint",
            },
        ],
        "notes": (
            "Code and Lean 4 formalization are released under the MIT License; documentation, "
            "result summaries, and the unreviewed technical supplement PDF are licensed under "
            "CC-BY-4.0; the bundled CDC 2017 YRBS dataset is U.S. Federal Government Public Domain."
        ),
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_staged_package(stage_dir: Path, mode: str) -> Dict[str, object]:
    """Run automated verification checks on the staged replication directory."""
    py_files = sorted(stage_dir.rglob("*.py"))
    py_errors: List[str] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        for idx, pf in enumerate(py_files):
            try:
                py_compile.compile(str(pf), cfile=str(tmp_path / f"mod_{idx}.pyc"), doraise=True)
            except py_compile.PyCompileError as exc:
                py_errors.append(f"{pf.relative_to(stage_dir)}: {exc}")

    # Lean 4 integrity check
    lean_dir = stage_dir / "lean" / "InattentivenessProofs"
    lean_files = sorted(lean_dir.glob("*.lean")) if lean_dir.is_dir() else []
    lean_issues: List[str] = []
    expected_lean = {
        "OrderedField.lean",
        "Basic.lean",
        "MeasurementModel.lean",
        "Fairness.lean",
        "CostBenefit.lean",
    }
    found_lean = {p.name for p in lean_files}
    if found_lean != expected_lean:
        lean_issues.append(f"Expected Lean modules {sorted(expected_lean)}, found {sorted(found_lean)}")
    for lf in lean_files:
        txt = lf.read_text(encoding="utf-8")
        if re.search(r"\bsorry\b", txt):
            lean_issues.append(f"Found `sorry` in {lf.relative_to(stage_dir)}")
        if re.search(r"\baxiom\b", txt):
            lean_issues.append(f"Found `axiom` in {lf.relative_to(stage_dir)}")

    # Secret / hardcoded path scan + blind anonymity scan on all text files
    text_exts = {
        ".py", ".md", ".txt", ".json", ".yaml", ".yml", ".sh", ".toml", ".lean", ".cff", ".ini", ".cfg",
    }
    secret_hits: List[str] = []
    blind_hits: List[str] = []
    for fp in sorted(stage_dir.rglob("*")):
        if not fp.is_file():
            continue
        rel = fp.relative_to(stage_dir)
        # Never allow env, encrypted credential files, or Python bytecode caches
        if (
            fp.name.startswith(".env")
            or fp.suffix == ".enc"
            or fp.suffix in SKIP_EXTENSIONS
            or "__pycache__" in rel.parts
        ):
            secret_hits.append(f"Forbidden file present: {rel}")
            continue
        if fp.suffix.lower() not in text_exts and fp.name not in {"LICENSE", "SHA256SUMS.txt"}:
            continue
        # Skip raw verification ledger JSON/MD lines that legitimately document the runner
        txt = fp.read_text(encoding="utf-8", errors="replace")
        for pat, desc in SECRET_PATTERNS:
            # Allow the descriptive prose mention "`/Users/...` paths" in verification/REPORT.md
            if rel.as_posix() == "results/verification/REPORT.md" and desc == "hardcoded macOS user path":
                continue
            if pat.search(txt):
                secret_hits.append(f"{rel}: matched {desc}")
        if mode == "blind":
            for pat, desc in BLIND_FORBIDDEN_PATTERNS:
                if pat.search(txt):
                    blind_hits.append(f"{rel}: leaked {desc}")

    # Required paper-evaluation harness completeness check (fail closed)
    missing_harness = [
        rel_req for rel_req in REQUIRED_HARNESS_FILES if not (stage_dir / rel_req).is_file()
    ]

    # Fail-closed PDF metadata & text check in blind mode
    supp_pdf = stage_dir / "docs" / "MISQ_online_supplement.pdf"
    if mode == "blind" and supp_pdf.is_file():
        blind_hits.extend(_inspect_pdf_for_blind(supp_pdf))

    report = {
        "python_files_compiled": len(py_files),
        "python_compile_errors": py_errors,
        "missing_harness_files": missing_harness,
        "lean_modules_checked": len(lean_files),
        "lean_issues": lean_issues,
        "secret_or_path_hits": secret_hits,
        "blind_anonymity_hits": blind_hits,
        "passed": not (py_errors or missing_harness or lean_issues or secret_hits or blind_hits),
    }
    return report


def _inspect_pdf_for_blind(pdf_path: Path) -> List[str]:
    """Fail-closed inspection of PDF text and metadata for blind-review leaks."""
    hits: List[str] = []
    rel_label = "docs/MISQ_online_supplement.pdf"

    # Defense-in-depth raw byte check for literal /Author metadata or forbidden tokens
    raw_bytes = pdf_path.read_bytes()
    raw_latin = raw_bytes.decode("latin1", errors="ignore")
    for m in re.finditer(r"/Author\s*(?:\(([^)]*)\)|<([0-9A-Fa-f]*)>)", raw_latin):
        val = (m.group(1) or m.group(2) or "").strip()
        if val:
            hits.append(f"{rel_label}: non-empty raw /Author metadata ({val!r})")
    for pat, desc in BLIND_FORBIDDEN_PATTERNS:
        if pat.search(raw_latin):
            hits.append(f"{rel_label}: raw PDF bytes leaked {desc}")

    # Path A: poppler-utils (pdftotext + pdfinfo)
    pdftotext_bin = shutil.which("pdftotext")
    pdfinfo_bin = shutil.which("pdfinfo")
    if pdftotext_bin and pdfinfo_bin:
        p_txt = subprocess.run(
            [pdftotext_bin, str(pdf_path), "-"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        p_info = subprocess.run(
            [pdfinfo_bin, str(pdf_path)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if p_txt.returncode != 0 or p_info.returncode != 0:
            hits.append(
                f"{rel_label}: PDF inspection failed (pdftotext rc={p_txt.returncode}, "
                f"pdfinfo rc={p_info.returncode}); refusing unverified PDF in blind mode"
            )
            return hits
        pdf_txt = p_txt.stdout.decode("utf-8", errors="replace")
        info_txt = p_info.stdout.decode("utf-8", errors="replace")
        for line in info_txt.splitlines():
            if line.lower().startswith("author:") and line.split(":", 1)[1].strip():
                hits.append(f"{rel_label}: non-empty PDF Author metadata ({line.strip()})")
        for pat, desc in BLIND_FORBIDDEN_PATTERNS:
            if pat.search(info_txt):
                hits.append(f"{rel_label}: PDF metadata leaked {desc}")
            if pat.search(pdf_txt):
                hits.append(f"{rel_label}: PDF text leaked {desc}")
        return hits

    # Path B: PyMuPDF fallback if poppler-utils is not installed
    try:
        import fitz  # type: ignore
    except ImportError:
        hits.append(
            f"{rel_label}: cannot verify PDF anonymity (neither poppler-utils [pdftotext+pdfinfo] "
            "nor pymupdf is available); install them or pass --omit-supplement-pdf"
        )
        return hits

    try:
        with fitz.open(str(pdf_path)) as doc:
            meta = doc.metadata or {}
            author_val = (meta.get("author") or "").strip()
            if author_val:
                hits.append(f"{rel_label}: non-empty PDF Author metadata ({author_val!r})")
            meta_blob = "\n".join(f"{k}: {v}" for k, v in meta.items() if v)
            page_txt = "\n".join(page.get_text("text") for page in doc)
    except Exception as exc:
        hits.append(f"{rel_label}: pymupdf failed to inspect PDF ({exc}); refusing unverified PDF in blind mode")
        return hits

    for pat, desc in BLIND_FORBIDDEN_PATTERNS:
        if pat.search(meta_blob):
            hits.append(f"{rel_label}: PDF metadata leaked {desc}")
        if pat.search(page_txt):
            hits.append(f"{rel_label}: PDF text leaked {desc}")
    return hits


def write_checksums_and_manifest(
    stage_dir: Path,
    mode: str,
    version: str,
    code_sha: Optional[str],
    exp_ref: Optional[str],
    exp_sha: Optional[str],
    paper_sha: Optional[str],
    hydration: Dict[str, object],
    sanitization_notes: List[str],
    verification: Dict[str, object],
) -> None:
    """Write MANIFEST.json and SHA256SUMS.txt covering every file in stage_dir."""
    # First write MANIFEST.json (without itself in the file list), then hash everything including MANIFEST.json.
    files_info = []
    total_bytes = 0
    for fp in sorted(stage_dir.rglob("*")):
        if not fp.is_file() or fp.name in {"MANIFEST.json", "SHA256SUMS.txt"}:
            continue
        rel = fp.relative_to(stage_dir).as_posix()
        size = fp.stat().st_size
        total_bytes += size
        files_info.append({"path": rel, "bytes": size, "sha256": sha256_file(fp)})

    manifest = {
        "package_name": stage_dir.name,
        "paper_title": PAPER_TITLE,
        "version": version,
        "mode": mode,
        "built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_commits": (
            {"redacted_for_blind_review": True}
            if mode == "blind"
            else {
                "autoencoders_census_main": code_sha,
                "autoencoders_census_experiment_ref": exp_ref,
                "autoencoders_census_experiment_sha": exp_sha,
                "inattentiveness_paper_head": paper_sha,
            }
        ),
        "hydration": (
            {
                "hydrated": bool(hydration.get("hydrated")),
                "cache_files_copied": hydration.get("cache_files_copied", 0),
                "derived_csvs_copied": hydration.get("derived_csvs_copied", []),
            }
            if mode == "blind"
            else hydration
        ),
        "sanitization_notes": sanitization_notes,
        "verification": verification,
        "file_count": len(files_info),
        "total_bytes": total_bytes,
        "files": files_info,
    }
    manifest_path = stage_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    checksum_lines = [f"{item['sha256']}  {item['path']}" for item in files_info]
    checksum_lines.append(f"{sha256_file(manifest_path)}  MANIFEST.json")
    checksum_lines.sort(key=lambda line: line.split("  ", 1)[1])
    (stage_dir / "SHA256SUMS.txt").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")


def create_archives(stage_dir: Path, out_dir: Path) -> Tuple[Path, Path]:
    """Create deterministic .zip and .tar.gz archives of stage_dir."""
    zip_path = out_dir / f"{stage_dir.name}.zip"
    tar_path = out_dir / f"{stage_dir.name}.tar.gz"
    all_files = sorted(fp for fp in stage_dir.rglob("*") if fp.is_file())

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for fp in all_files:
            arcname = f"{stage_dir.name}/{fp.relative_to(stage_dir).as_posix()}"
            zf.write(fp, arcname=arcname)

    with tarfile.open(tar_path, "w:gz", compresslevel=6) as tf:
        for fp in all_files:
            arcname = f"{stage_dir.name}/{fp.relative_to(stage_dir).as_posix()}"
            tf.add(fp, arcname=arcname, recursive=False)

    # Round-trip integrity verification
    with zipfile.ZipFile(zip_path, "r") as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"Corrupt entry in zip archive: {bad}")
    with tarfile.open(tar_path, "r:gz") as tf:
        if len(tf.getnames()) != len(all_files):
            raise RuntimeError("Tarball member count mismatch")

    return zip_path, tar_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build a verified Zenodo replication package.")
    parser.add_argument(
        "--mode",
        choices=["camera-ready", "blind"],
        default="camera-ready",
        help="camera-ready (full Zenodo metadata) or blind (double-anonymous review archive)",
    )
    parser.add_argument("--version", default="1.0.0", help="Semantic version tag (default: 1.0.0)")
    parser.add_argument(
        "--code-repo",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Path to autoencoders_census repository",
    )
    parser.add_argument(
        "--paper-repo",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent / "inattentiveness_paper",
        help="Path to inattentiveness_paper repository",
    )
    parser.add_argument(
        "--experiment-ref",
        default=None,
        help="Optional git ref in autoencoders_census providing evaluate/detection.py + experiment_b/ + revision_experiments/",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=None,
        help="Optional path to restored 2026-10-23 paper-data snapshot to hydrate score caches (code/cache/)",
    )
    parser.add_argument(
        "--include-third-party-raw-data",
        action="store_true",
        help="Also copy the 8 external benchmark CSVs from --snapshot-dir/data/ (verify per-source licenses first!)",
    )
    parser.add_argument(
        "--include-weights",
        action="store_true",
        help="When --snapshot-dir is supplied, also copy heavy TensorFlow SavedModel weight directories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <code-repo>/dist/zenodo)",
    )
    parser.add_argument(
        "--omit-supplement-pdf",
        action="store_true",
        help="Do not bundle docs/MISQ_online_supplement.pdf (useful in blind mode on hosts without PDF tools)",
    )
    parser.add_argument("--skip-archives", action="store_true", help="Stage directory only; do not write .zip/.tar.gz")
    args = parser.parse_args(argv)

    code_repo = args.code_repo.resolve()
    paper_repo = args.paper_repo.resolve()
    out_dir = (args.output_dir or (code_repo / "dist" / "zenodo")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = "-blind" if args.mode == "blind" else ""
    pkg_name = f"inattentiveness-screening-replication{suffix}-v{args.version}"
    stage_dir = out_dir / pkg_name
    # Always invalidate previous archives when replacing the staged directory so
    # --skip-archives or a failed rebuild can never leave stale ZIP/tar artifacts
    # paired with a newer MANIFEST.json / SHA256SUMS.txt.
    for stale_path in (
        out_dir / f"{pkg_name}.zip",
        out_dir / f"{pkg_name}.tar.gz",
        out_dir / f"{pkg_name}.zenodo.json",
    ):
        if stale_path.exists():
            stale_path.unlink()
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True)

    code_root = stage_dir / "code"
    code_root.mkdir()

    # 1. Copy core files and directories from autoencoders_census working tree
    for rel in MAIN_CODE_FILES:
        src = code_repo / rel
        if src.is_file():
            dst = code_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    for rel_dir in MAIN_CODE_DIRS:
        src = code_repo / rel_dir
        if src.is_dir():
            copy_tree_clean(src, code_root / rel_dir)

    for rel_test in PURE_ML_TESTS:
        src = code_repo / rel_test
        if src.is_file():
            dst = code_root / rel_test
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    if (code_repo / "requirements.txt").is_file():
        shutil.copy2(code_repo / "requirements.txt", code_root / "requirements-full.txt")
    write_requirements_replication(code_root)

    # 2. Overlay the verified paper evaluation harness & experiment scripts
    exp_ref, exp_sha = resolve_experiment_ref(code_repo, args.experiment_ref)
    if exp_sha:
        for rel in OVERLAY_PATHS:
            extract_git_path(code_repo, exp_sha, rel, code_root)
    else:
        for rel in ("dataset", "model", "evaluate", "experiment_b", "revision_experiments"):
            if (code_repo / rel).is_dir():
                copy_tree_clean(code_repo / rel, code_root / rel)
        if (code_repo / "utils.py").is_file():
            shutil.copy2(code_repo / "utils.py", code_root / "utils.py")

    # 3. Copy public-domain CDC 2017 YRBS dataset
    (code_root / "data").mkdir(exist_ok=True)
    for rel in PUBLIC_DOMAIN_DATA_FILES:
        src = code_repo / rel
        if src.is_file():
            shutil.copy2(src, code_root / rel)

    # 4. Copy verified aggregate results + verification ledger
    results_dir = stage_dir / "results"
    results_dir.mkdir()
    rev_results = code_repo / "scripts" / "misq_revision" / "results"
    if rev_results.is_dir():
        for jf in sorted(rev_results.glob("*.json")):
            shutil.copy2(jf, results_dir / jf.name)

    verif_src = paper_repo / "facts-and-figures-out"
    if verif_src.is_dir():
        verif_dst = results_dir / "verification"
        verif_dst.mkdir()
        for fname in ("REPORT.md", "results.csv", "verification-report.json"):
            if (verif_src / fname).is_file():
                shutil.copy2(verif_src / fname, verif_dst / fname)
        if args.mode == "blind":
            sanitize_verification_for_blind(verif_dst)

    # 5. Copy zero-dependency Lean 4 formalization
    lean_src = paper_repo / "lean"
    if not lean_src.is_dir():
        raise FileNotFoundError(f"Lean 4 directory not found at {lean_src}")
    copy_tree_clean(lean_src, stage_dir / "lean")

    # 6. Copy unreviewed technical supplement PDF (unless --omit-supplement-pdf)
    docs_dir = stage_dir / "docs"
    docs_dir.mkdir()
    supp_pdf = paper_repo / "MISQ_online_supplement.pdf"
    if not args.omit_supplement_pdf and supp_pdf.is_file():
        shutil.copy2(supp_pdf, docs_dir / "MISQ_online_supplement.pdf")

    # 7. Optional hydration from restored 2026-10-23 paper-data snapshot
    hydration: Dict[str, object] = {
        "hydrated": False,
        "cache_files_copied": 0,
        "derived_csvs_copied": [],
        "raw_csvs_copied": [],
        "weights_included": False,
    }
    if args.snapshot_dir:
        snap = args.snapshot_dir.resolve()
        if not snap.is_dir():
            raise FileNotFoundError(f"--snapshot-dir does not exist: {snap}")
        hydration = hydrate_from_snapshot(
            snap,
            code_root,
            include_third_party_raw=args.include_third_party_raw_data,
            include_weights=args.include_weights,
        )

    # 8. Sanitize hardcoded developer paths (and repo links in blind mode)
    sanitization_notes = sanitize_code_paths(code_root, blind=(args.mode == "blind"))

    # 9. Write root documentation, licenses, and metadata
    (stage_dir / "README.md").write_text(
        build_readme(args.mode, args.version, hydration, pkg_name), encoding="utf-8"
    )
    (stage_dir / "REPRODUCTION_MAP.md").write_text(build_reproduction_map(), encoding="utf-8")
    (stage_dir / "DATA_PROVENANCE.md").write_text(build_data_provenance(hydration), encoding="utf-8")
    (stage_dir / "LICENSE").write_text(build_mit_license(args.mode), encoding="utf-8")
    (stage_dir / "LICENSE-DOCS.txt").write_text(build_docs_license(), encoding="utf-8")

    if args.mode == "camera-ready":
        (stage_dir / "CITATION.cff").write_text(build_citation_cff(args.version), encoding="utf-8")
        zenodo_meta = build_zenodo_json(args.version)
        zenodo_str = json.dumps(zenodo_meta, indent=2) + "\n"
        (stage_dir / ".zenodo.json").write_text(zenodo_str, encoding="utf-8")
        (out_dir / f"{pkg_name}.zenodo.json").write_text(zenodo_str, encoding="utf-8")

    # 10. Automated verification suite
    verification = verify_staged_package(stage_dir, args.mode)

    # 11. Write MANIFEST.json and SHA256SUMS.txt
    write_checksums_and_manifest(
        stage_dir=stage_dir,
        mode=args.mode,
        version=args.version,
        code_sha=git_rev_parse(code_repo, "HEAD"),
        exp_ref=exp_ref,
        exp_sha=exp_sha,
        paper_sha=git_rev_parse(paper_repo, "HEAD"),
        hydration=hydration,
        sanitization_notes=sanitization_notes,
        verification=verification,
    )

    if not verification["passed"]:
        stale_meta = out_dir / f"{pkg_name}.zenodo.json"
        if stale_meta.exists():
            stale_meta.unlink()
        print("[FAIL] Verification checks failed:", file=sys.stderr)
        print(json.dumps(verification, indent=2), file=sys.stderr)
        if verification.get("missing_harness_files"):
            print(
                "[HINT] Fetch the paper evaluation harness branch first:\n"
                "       git fetch origin revision/experiment-scripts\n"
                "       or pass --experiment-ref <git-ref>.",
                file=sys.stderr,
            )
        return 1

    print(f"[OK] Staged {pkg_name} -> {stage_dir}")
    print(
        f"     Python files compiled: {verification['python_files_compiled']}, "
        f"Lean modules verified: {verification['lean_modules_checked']} (0 sorry, 0 axiom), "
        f"Secrets/path hits: 0, Blind leaks: {len(verification['blind_anonymity_hits'])}"
    )

    if not args.skip_archives:
        zip_path, tar_path = create_archives(stage_dir, out_dir)
        zip_sha = sha256_file(zip_path)
        tar_sha = sha256_file(tar_path)
        print(f"[OK] ZIP:    {zip_path} ({zip_path.stat().st_size / (1024*1024):.2f} MiB, sha256:{zip_sha[:16]})")
        print(f"[OK] TAR.GZ: {tar_path} ({tar_path.stat().st_size / (1024*1024):.2f} MiB, sha256:{tar_sha[:16]})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
