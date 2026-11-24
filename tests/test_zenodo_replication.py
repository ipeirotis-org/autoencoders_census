"""Unit tests for the Zenodo replication package builder and uploader."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from replication import build_zenodo_package as bzp
from replication import upload_to_zenodo as utz


def _make_stub_repos(tmp: Path) -> tuple[Path, Path, Path]:
    """Create minimal synthetic code, paper, and snapshot dirs for isolated tests."""
    code_repo = tmp / "code_repo"
    paper_repo = tmp / "paper_repo"
    snap_dir = tmp / "snapshot"

    # Minimal code repo layout
    for rel in (
        "main.py",
        "chow_liu_rank.py",
        "utils.py",
        "dataset/__init__.py",
        "dataset/loader.py",
        "features/__init__.py",
        "features/transform.py",
        "model/__init__.py",
        "model/autoencoder.py",
        "evaluate/__init__.py",
        "evaluate/detection.py",
        "train/__init__.py",
        "config/simple_autoencoder.yaml",
        "scripts/misq_revision/run_coherence_sim.py",
        "scripts/misq_revision/results/coherence_sim_results.json",
        "tests/__init__.py",
        "tests/test_chow_liu.py",
        "data/sadc_2017only_national_full.csv",
        "requirements.txt",
    ):
        fp = code_repo / rel
        fp.parent.mkdir(parents=True, exist_ok=True)
        if fp.suffix == ".py":
            fp.write_text("# stub module\nVALUE = 1\n", encoding="utf-8")
        elif fp.suffix == ".json":
            fp.write_text('{"ok": true}\n', encoding="utf-8")
        elif fp.suffix == ".csv":
            fp.write_text("col1,col2\n1,2\n", encoding="utf-8")
        else:
            fp.write_text("numpy==1.26.4\n", encoding="utf-8")

    # Include a README with identifying refs to verify blind-mode sanitization
    rev_readme = code_repo / "scripts" / "misq_revision" / "README.md"
    rev_readme.write_text(
        "(companion PR [`ipeirotis-org/inattentiveness_paper#32`](https://github.com/ipeirotis-org/inattentiveness_paper/pull/32)) "
        "using `gs://autoencoders-census-paper-data/snapshots/2026-10-23`.\n",
        encoding="utf-8",
    )

    # Minimal paper repo with Lean 4 modules, verification ledger, and stub PDF
    lean_mods = [
        "OrderedField.lean",
        "Basic.lean",
        "MeasurementModel.lean",
        "Fairness.lean",
        "CostBenefit.lean",
    ]
    for mod in lean_mods:
        lf = paper_repo / "lean" / "InattentivenessProofs" / mod
        lf.parent.mkdir(parents=True, exist_ok=True)
        lf.write_text(f"theorem stub_{mod[:-5]} : True := trivial\n", encoding="utf-8")
    (paper_repo / "lean" / "InattentivenessProofs.lean").write_text("-- root\n", encoding="utf-8")
    (paper_repo / "lean" / "lean-toolchain").write_text("leanprover/lean4:v4.35.0\n", encoding="utf-8")
    (paper_repo / "lean" / "lakefile.toml").write_text('name = "inattentiveness_proofs"\n', encoding="utf-8")
    (paper_repo / "lean" / "lake-manifest.json").write_text('{"version": "1.1.0", "packages": []}\n', encoding="utf-8")
    (paper_repo / "lean" / "README.md").write_text("# Lean proofs\n", encoding="utf-8")

    verif_dir = paper_repo / "facts-and-figures-out"
    verif_dir.mkdir(parents=True, exist_ok=True)
    (verif_dir / "REPORT.md").write_text(
        "Restored from `gs://autoencoders-census-paper-data/snapshots/2026-10-23`.\n",
        encoding="utf-8",
    )
    (verif_dir / "results.csv").write_text("id,status\nE01,match\n", encoding="utf-8")
    (verif_dir / "verification-report.json").write_text(
        '{"bucket": "gs://autoencoders-census-paper-data/snapshots/2026-10-23"}\n',
        encoding="utf-8",
    )
    (paper_repo / "MISQ_online_supplement.pdf").write_bytes(b"%PDF-1.4\n%stub\n%%EOF\n")

    # Minimal snapshot for hydration test (score caches + heavy weights + third-party CSV)
    (snap_dir / "cache" / "_fix9_inattentive_cl").mkdir(parents=True, exist_ok=True)
    (snap_dir / "cache" / "_fix9_inattentive_cl" / "errors.csv").write_text("row,pct\n0,0.5\n", encoding="utf-8")
    (snap_dir / "cache" / "_tuned_inattentive_ae" / "autoencoder").mkdir(parents=True, exist_ok=True)
    (snap_dir / "cache" / "_tuned_inattentive_ae" / "best_hyperparameters.yaml").write_text("latent_space_dim: 4\n", encoding="utf-8")
    (snap_dir / "cache" / "_tuned_inattentive_ae" / "autoencoder" / "saved_model.pb").write_bytes(b"heavy-weights")
    (snap_dir / "data").mkdir(parents=True, exist_ok=True)
    (snap_dir / "data" / "inattentive_users.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    return code_repo, paper_repo, snap_dir


class TestZenodoReplicationPackage(unittest.TestCase):
    def test_camera_ready_and_blind_builds_pass_all_checks(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            code_repo, paper_repo, snap_dir = _make_stub_repos(tmp)
            out_dir = tmp / "out"

            # 1. Camera-ready build with snapshot score-cache hydration (no raw CSVs / weights)
            rc = bzp.main([
                "--mode", "camera-ready",
                "--version", "1.0.0",
                "--code-repo", str(code_repo),
                "--paper-repo", str(paper_repo),
                "--snapshot-dir", str(snap_dir),
                "--output-dir", str(out_dir),
            ])
            self.assertEqual(rc, 0)

            stage = out_dir / "inattentiveness-screening-replication-v1.0.0"
            self.assertTrue((stage / ".zenodo.json").is_file())
            self.assertTrue((stage / "CITATION.cff").is_file())
            self.assertTrue((stage / "code" / "cache" / "_fix9_inattentive_cl" / "errors.csv").is_file())
            self.assertTrue((stage / "code" / "cache" / "_tuned_inattentive_ae" / "best_hyperparameters.yaml").is_file())
            # Heavy weights and third-party raw CSVs stay excluded by default
            self.assertFalse((stage / "code" / "cache" / "_tuned_inattentive_ae" / "autoencoder" / "saved_model.pb").exists())
            self.assertFalse((stage / "code" / "data" / "inattentive_users.csv").exists())
            # Zero bytecode files leaked into staged tree
            self.assertEqual(list(stage.rglob("*.pyc")), [])

            # Verify SHA256SUMS.txt matches every single file
            for line in (stage / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
                expected_hash, rel_path = line.split("  ", 1)
                actual_hash = hashlib.sha256((stage / rel_path).read_bytes()).hexdigest()
                self.assertEqual(actual_hash, expected_hash, f"Checksum mismatch for {rel_path}")

            # Verify zip archive round-trip
            zip_path = out_dir / "inattentiveness-screening-replication-v1.0.0.zip"
            self.assertTrue(zip_path.is_file())
            with zipfile.ZipFile(zip_path, "r") as zf:
                self.assertIsNone(zf.testzip())

            # 2. Blind build strips author metadata and passes zero-leak audit
            rc_blind = bzp.main([
                "--mode", "blind",
                "--version", "1.0.0",
                "--code-repo", str(code_repo),
                "--paper-repo", str(paper_repo),
                "--output-dir", str(out_dir),
            ])
            self.assertEqual(rc_blind, 0)

            blind_stage = out_dir / "inattentiveness-screening-replication-blind-v1.0.0"
            self.assertFalse((blind_stage / ".zenodo.json").exists())
            self.assertFalse((blind_stage / "CITATION.cff").exists())
            manifest = json.loads((blind_stage / "MANIFEST.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["source_commits"].get("redacted_for_blind_review"))
            self.assertTrue(manifest["verification"]["passed"])
            self.assertEqual(manifest["verification"]["blind_anonymity_hits"], [])

    def test_lean_sorry_fails_verification(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            code_repo, paper_repo, _ = _make_stub_repos(tmp)
            (paper_repo / "lean" / "InattentivenessProofs" / "Fairness.lean").write_text(
                "theorem bad : True := by sorry\n", encoding="utf-8"
            )
            rc = bzp.main([
                "--mode", "camera-ready",
                "--code-repo", str(code_repo),
                "--paper-repo", str(paper_repo),
                "--output-dir", str(tmp / "out"),
                "--skip-archives",
            ])
            self.assertEqual(rc, 1)

    def test_path_sanitization_rewrites_hardcoded_developer_paths(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            code_root = Path(td)
            p1 = code_root / "experiment_b" / "pl_tradeoff_figs.py"
            p1.parent.mkdir(parents=True)
            p1.write_text(
                'import os\nFIG = "/Users/iliastriantafyllopoulos/Documents/my_projects/inattentiveness_paper/figures"\n',
                encoding="utf-8",
            )
            notes = bzp.sanitize_code_paths(code_root, blind=True)
            self.assertEqual(len(notes), 1)
            rewritten = p1.read_text(encoding="utf-8")
            self.assertNotIn("/Users/", rewritten)
            self.assertIn('os.environ.get("FIG_DIR"', rewritten)

    def test_upload_publish_guard_blocks_without_explicit_confirmation(self) -> None:
        rc = utz.main(["--package-dir", "/nonexistent", "--publish"])
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
