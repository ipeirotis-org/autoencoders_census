"""Unit tests for the Zenodo replication package builder and uploader."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from replication import build_zenodo_package as bzp
from replication import upload_to_zenodo as utz


def _make_minimal_valid_pdf(author: str = "") -> bytes:
    """Create a tiny standards-compliant 1-page PDF with optional /Author metadata."""
    objects = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] >>\nendobj\n",
        f"4 0 obj\n<< /Author ({author}) /Title (Supplement) >>\nendobj\n".encode("latin1"),
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(len(out))
        out.extend(obj)
    xref_pos = len(out)
    out.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii"))
    for off in offsets[1:]:
        out.extend(f"{off:010d} 00000 n \n".encode("ascii"))
    out.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R /Info 4 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode("ascii")
    )
    return bytes(out)


def _make_stub_repos(tmp: Path, include_harness: bool = True, pdf_author: str = "") -> tuple[Path, Path, Path]:
    """Create minimal synthetic code, paper, and snapshot dirs for isolated tests."""
    code_repo = tmp / "code_repo"
    paper_repo = tmp / "paper_repo"
    snap_dir = tmp / "snapshot"

    base_files = [
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
        "train/__init__.py",
        "config/simple_autoencoder.yaml",
        "scripts/misq_revision/run_coherence_sim.py",
        "scripts/misq_revision/run_all_snap_experiments.py",
        "scripts/misq_revision/generate_snap_figures.py",
        "scripts/misq_revision/results/coherence_sim_results.json",
        "tests/__init__.py",
        "tests/test_chow_liu.py",
        "data/sadc_2017only_national_full.csv",
        "requirements.txt",
    ]
    if include_harness:
        base_files.extend([
            "evaluate/detection.py",
            "evaluate/baselines.py",
            "experiment_b/__init__.py",
            "experiment_b/fairness.py",
            "revision_experiments/lib_r2.py",
        ])

    for rel in base_files:
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

    # Give simulation/figure scripts the exact savefig lines tested for makedirs injection
    (code_repo / "scripts" / "misq_revision" / "run_coherence_sim.py").write_text(
        'import os\nOUT_FIG = "out.pdf"\ndef main(fig):\n    fig.savefig(OUT_FIG, bbox_inches="tight")\n',
        encoding="utf-8",
    )
    (code_repo / "scripts" / "misq_revision" / "generate_snap_figures.py").write_text(
        'import os\nPAPER_DIR = "."\nFIG_DIR = os.environ.get("FIG_DIR", os.path.join(PAPER_DIR, "figures"))\n',
        encoding="utf-8",
    )

    # Include a README with identifying refs to verify blind-mode sanitization
    rev_readme = code_repo / "scripts" / "misq_revision" / "README.md"
    rev_readme.write_text(
        "(companion PR [`ipeirotis-org/inattentiveness_paper#32`](https://github.com/ipeirotis-org/inattentiveness_paper/pull/32)) "
        "using `gs://autoencoders-census-paper-data/snapshots/2026-10-23`.\n",
        encoding="utf-8",
    )

    # Minimal paper repo with Lean 4 modules, verification ledger, and minimal valid PDF
    for mod in (
        "OrderedField.lean",
        "Basic.lean",
        "MeasurementModel.lean",
        "Fairness.lean",
        "CostBenefit.lean",
    ):
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
    (paper_repo / "MISQ_online_supplement.pdf").write_bytes(_make_minimal_valid_pdf(author=pdf_author))

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
            # Quick-start README and staged simulation scripts create ../outputs/figures
            readme_txt = (stage / "README.md").read_text(encoding="utf-8")
            self.assertIn("mkdir -p ../outputs/figures", readme_txt)
            sim_txt = (stage / "code" / "scripts" / "misq_revision" / "run_coherence_sim.py").read_text(encoding="utf-8")
            self.assertIn("os.makedirs(os.path.dirname(os.path.abspath(OUT_FIG)), exist_ok=True)", sim_txt)
            snap_fig_txt = (stage / "code" / "scripts" / "misq_revision" / "generate_snap_figures.py").read_text(encoding="utf-8")
            self.assertIn("os.makedirs(FIG_DIR, exist_ok=True)", snap_fig_txt)

            # Verify SHA256SUMS.txt matches every single file
            for line in (stage / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
                expected_hash, rel_path = line.split("  ", 1)
                actual_hash = hashlib.sha256((stage / rel_path).read_bytes()).hexdigest()
                self.assertEqual(actual_hash, expected_hash, f"Checksum mismatch for {rel_path}")

            # Verify zip archive round-trip and freshness check
            zip_path = out_dir / "inattentiveness-screening-replication-v1.0.0.zip"
            self.assertTrue(zip_path.is_file())
            with zipfile.ZipFile(zip_path, "r") as zf:
                self.assertIsNone(zf.testzip())
            utz.verify_archive_matches_stage(stage, zip_path)

            # 2. Blind build with --omit-supplement-pdf works on any host (even without poppler)
            rc_blind = bzp.main([
                "--mode", "blind",
                "--version", "1.0.0",
                "--code-repo", str(code_repo),
                "--paper-repo", str(paper_repo),
                "--omit-supplement-pdf",
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

    def test_missing_harness_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            code_repo, paper_repo, _ = _make_stub_repos(tmp, include_harness=False)
            out_dir = tmp / "out"
            rc = bzp.main([
                "--mode", "camera-ready",
                "--code-repo", str(code_repo),
                "--paper-repo", str(paper_repo),
                "--output-dir", str(out_dir),
            ])
            self.assertEqual(rc, 1)
            self.assertFalse((out_dir / "inattentiveness-screening-replication-v1.0.0.zip").exists())
            manifest = json.loads(
                (out_dir / "inattentiveness-screening-replication-v1.0.0" / "MANIFEST.json").read_text(encoding="utf-8")
            )
            self.assertIn("code/evaluate/detection.py", manifest["verification"]["missing_harness_files"])

    def test_blind_pdf_fails_closed_on_author_metadata_or_missing_tools(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            code_repo, paper_repo, _ = _make_stub_repos(tmp, pdf_author="Jane Doe")
            out_dir = tmp / "out"
            # Non-empty /Author metadata must fail closed regardless of installed PDF binaries
            rc_author = bzp.main([
                "--mode", "blind",
                "--code-repo", str(code_repo),
                "--paper-repo", str(paper_repo),
                "--output-dir", str(out_dir),
                "--skip-archives",
            ])
            self.assertEqual(rc_author, 1)

            # Clean PDF but NO pdftotext/pdfinfo and NO pymupdf must ALSO fail closed
            (paper_repo / "MISQ_online_supplement.pdf").write_bytes(_make_minimal_valid_pdf(author=""))
            with mock.patch("replication.build_zenodo_package.shutil.which", return_value=None):
                with mock.patch.dict(sys.modules, {"fitz": None}):
                    rc_notools = bzp.main([
                        "--mode", "blind",
                        "--code-repo", str(code_repo),
                        "--paper-repo", str(paper_repo),
                        "--output-dir", str(out_dir),
                        "--skip-archives",
                    ])
            self.assertEqual(rc_notools, 1)

    def test_rebuild_skip_archives_invalidates_stale_zip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            code_repo, paper_repo, _ = _make_stub_repos(tmp)
            out_dir = tmp / "out"
            rc1 = bzp.main([
                "--mode", "camera-ready",
                "--code-repo", str(code_repo),
                "--paper-repo", str(paper_repo),
                "--output-dir", str(out_dir),
            ])
            self.assertEqual(rc1, 0)
            zip_path = out_dir / "inattentiveness-screening-replication-v1.0.0.zip"
            tar_path = out_dir / "inattentiveness-screening-replication-v1.0.0.tar.gz"
            self.assertTrue(zip_path.is_file())
            self.assertTrue(tar_path.is_file())

            # Rebuild with --skip-archives: previous ZIP and tar.gz must be removed
            rc2 = bzp.main([
                "--mode", "camera-ready",
                "--code-repo", str(code_repo),
                "--paper-repo", str(paper_repo),
                "--output-dir", str(out_dir),
                "--skip-archives",
            ])
            self.assertEqual(rc2, 0)
            self.assertFalse(zip_path.exists())
            self.assertFalse(tar_path.exists())

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

    def test_upload_publish_guard_and_existing_deposition_reuse(self) -> None:
        # 1. Guard blocks --publish without explicit confirmation flag
        rc = utz.main(["--package-dir", "/nonexistent", "--publish"])
        self.assertEqual(rc, 2)

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            code_repo, paper_repo, _ = _make_stub_repos(tmp)
            out_dir = tmp / "out"
            self.assertEqual(
                bzp.main([
                    "--mode", "camera-ready",
                    "--code-repo", str(code_repo),
                    "--paper-repo", str(paper_repo),
                    "--output-dir", str(out_dir),
                ]),
                0,
            )
            stage = out_dir / "inattentiveness-screening-replication-v1.0.0"
            zip_path = out_dir / "inattentiveness-screening-replication-v1.0.0.zip"

            # 2. Reuse an existing reserved draft via --deposition-id (must NOT POST new draft)
            calls: list[tuple[str, str]] = []

            def fake_api(method: str, url: str, token: str, payload=None):
                calls.append((method, url))
                if method == "GET":
                    return {"id": 98765, "submitted": False}
                if method == "PUT":
                    return {
                        "id": 98765,
                        "links": {"bucket": "https://zenodo.org/api/files/b1", "html": "https://zenodo.org/deposit/98765"},
                        "metadata": {"prereserve_doi": {"doi": "10.5281/zenodo.98765"}},
                    }
                if method == "POST" and url.endswith("/actions/publish"):
                    return {
                        "doi": "10.5281/zenodo.98765",
                        "links": {"record_html": "https://zenodo.org/records/98765"},
                    }
                raise AssertionError(f"Unexpected API call: {method} {url}")

            uploaded: list[str] = []
            with mock.patch.dict(os.environ, {"ZENODO_TOKEN": "fake-token"}):
                with mock.patch("replication.upload_to_zenodo.api_json", side_effect=fake_api):
                    with mock.patch(
                        "replication.upload_to_zenodo.upload_file_to_bucket",
                        side_effect=lambda bucket, fp, tok: uploaded.append(fp.name) or {},
                    ):
                        rc_pub = utz.main([
                            "--production",
                            "--deposition-id", "98765",
                            "--publish",
                            "--yes-i-know-this-is-public-and-permanent",
                            "--package-dir", str(stage),
                        ])
            self.assertEqual(rc_pub, 0)
            self.assertEqual(
                calls,
                [
                    ("GET", "https://zenodo.org/api/deposit/depositions/98765"),
                    ("PUT", "https://zenodo.org/api/deposit/depositions/98765"),
                    ("POST", "https://zenodo.org/api/deposit/depositions/98765/actions/publish"),
                ],
            )
            self.assertEqual(uploaded, [zip_path.name, "SHA256SUMS.txt", "README.md"])

            # 3. Stale ZIP mismatch against staged MANIFEST.json is rejected before upload
            (stage / "MANIFEST.json").write_text(
                (stage / "MANIFEST.json").read_text(encoding="utf-8") + "\n", encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                utz.verify_archive_matches_stage(stage, zip_path)


if __name__ == "__main__":
    unittest.main()
