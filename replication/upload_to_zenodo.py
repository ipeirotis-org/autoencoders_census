#!/usr/bin/env python3
"""Create, update, or publish a Zenodo deposit from a built replication package
without accidentally minting a public DOI before paper acceptance or stranding a
previously reserved DOI.

Safety defaults:
  - Targets https://sandbox.zenodo.org by default (pass --production for zenodo.org).
  - Creates a PRIVATE DRAFT with a prereserved DOI; NEVER publishes unless both
    --publish and --yes-i-know-this-is-public-and-permanent are explicitly passed.
  - Supports --deposition-id <ID> to update and publish a previously reserved draft
    so its prereserved DOI becomes the live published DOI.
  - Verifies that the ZIP archive matches the staged MANIFEST.json and passed all
    verification checks before uploading anything.
  - Reads the API token strictly from ZENODO_TOKEN (never written to disk).

Zero external dependencies: standard-library Python 3.10+.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional


SANDBOX_BASE = "https://sandbox.zenodo.org/api"
PROD_BASE = "https://zenodo.org/api"


def api_json(
    method: str,
    url: str,
    token: str,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Zenodo API {method} {url} failed ({exc.code}): {body}") from exc


def upload_file_to_bucket(bucket_url: str, file_path: Path, token: str) -> Dict[str, Any]:
    url = f"{bucket_url.rstrip('/')}/{file_path.name}"
    file_bytes = file_path.read_bytes()
    req = urllib.request.Request(
        url,
        data=file_bytes,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/octet-stream",
            "Content-Length": str(len(file_bytes)),
        },
        method="PUT",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Bucket PUT {file_path.name} failed ({exc.code}): {body}") from exc


def verify_archive_matches_stage(stage_dir: Path, zip_file: Path) -> None:
    """Ensure the ZIP archive is in sync with the staged MANIFEST.json and verified."""
    manifest_path = stage_dir / "MANIFEST.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing MANIFEST.json in {stage_dir}")
    staged_bytes = manifest_path.read_bytes()
    manifest = json.loads(staged_bytes.decode("utf-8"))
    if not manifest.get("verification", {}).get("passed", False):
        raise ValueError("Staged MANIFEST.json reports verification.passed != true; rebuild before uploading.")

    expected_member = f"{stage_dir.name}/MANIFEST.json"
    with zipfile.ZipFile(zip_file, "r") as zf:
        if expected_member not in zf.namelist():
            raise ValueError(f"Archive {zip_file.name} is missing {expected_member}; rebuild archives.")
        archived_bytes = zf.read(expected_member)
    if archived_bytes != staged_bytes:
        raise ValueError(
            f"Archive {zip_file.name} does not match staged {manifest_path}; rebuild without --skip-archives."
        )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Create, update, or publish a Zenodo deposit from a built package.")
    parser.add_argument(
        "--package-dir",
        type=Path,
        required=True,
        help="Path to the staged camera-ready directory (e.g. dist/zenodo/inattentiveness-screening-replication-v1.0.0)",
    )
    parser.add_argument(
        "--deposition-id",
        default=None,
        help="Existing Zenodo draft deposition ID to update/publish (reuses its prereserved DOI)",
    )
    parser.add_argument(
        "--production",
        action="store_true",
        help="Target production zenodo.org instead of sandbox.zenodo.org",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Publish the deposit immediately (ONLY use upon acceptance; breaks double-blind review!)",
    )
    parser.add_argument(
        "--yes-i-know-this-is-public-and-permanent",
        action="store_true",
        help="Required confirmation flag alongside --publish",
    )
    args = parser.parse_args(argv)

    if args.publish and not args.yes_i_know_this_is_public_and_permanent:
        print(
            "[ABORT] Publishing on Zenodo mints a public, permanent DOI with author names.\n"
            "        To preserve MIS Quarterly double-anonymous review, keep the deposit in DRAFT\n"
            "        (omit --publish) until conditional acceptance, OR re-run with\n"
            "        --publish --yes-i-know-this-is-public-and-permanent.",
            file=sys.stderr,
        )
        return 2

    stage_dir = args.package_dir.resolve()
    meta_file = stage_dir / ".zenodo.json"
    zip_file = stage_dir.parent / f"{stage_dir.name}.zip"
    sums_file = stage_dir / "SHA256SUMS.txt"
    readme_file = stage_dir / "README.md"

    for req_file in (meta_file, zip_file, sums_file, readme_file):
        if not req_file.is_file():
            print(f"[ERROR] Missing required artifact: {req_file}", file=sys.stderr)
            return 1

    try:
        verify_archive_matches_stage(stage_dir, zip_file)
    except (ValueError, zipfile.BadZipFile) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    token = os.environ.get("ZENODO_TOKEN", "").strip()
    if not token:
        env_hint = "zenodo.org" if args.production else "sandbox.zenodo.org"
        print(
            f"[ERROR] ZENODO_TOKEN environment variable is not set.\n"
            f"        Create a personal access token with deposit:write (and deposit:actions if publishing)\n"
            f"        at https://{env_hint}/account/settings/applications/tokens/new/ and export ZENODO_TOKEN.",
            file=sys.stderr,
        )
        return 1

    metadata = json.loads(meta_file.read_text(encoding="utf-8"))
    metadata["prereserve_doi"] = True

    base = PROD_BASE if args.production else SANDBOX_BASE
    env_label = "PRODUCTION (zenodo.org)" if args.production else "SANDBOX (sandbox.zenodo.org)"

    if args.deposition_id:
        dep_id = str(args.deposition_id).strip()
        print(f"[1/4] Loading existing draft deposition {dep_id} on {env_label}...")
        dep = api_json("GET", f"{base}/deposit/depositions/{dep_id}", token)
        if dep.get("submitted"):
            print(
                f"[ERROR] Deposition {dep_id} is already published (submitted=true). "
                "Create a new version on Zenodo instead.",
                file=sys.stderr,
            )
            return 1
        dep = api_json("PUT", f"{base}/deposit/depositions/{dep_id}", token, payload={"metadata": metadata})
    else:
        print(f"[1/4] Creating new draft deposition on {env_label}...")
        dep = api_json("POST", f"{base}/deposit/depositions", token, payload={"metadata": metadata})
        dep_id = str(dep["id"])

    bucket_url = dep["links"]["bucket"]
    reserved_doi = dep.get("metadata", {}).get("prereserve_doi", {}).get("doi", "<unknown>")
    html_url = dep.get("links", {}).get("html", "")
    print(f"      Deposit ID:   {dep_id}")
    print(f"      Reserved DOI: {reserved_doi}")

    print(f"[2/4] Uploading {zip_file.name} ({zip_file.stat().st_size / (1024*1024):.2f} MiB)...")
    upload_file_to_bucket(bucket_url, zip_file, token)
    print(f"[3/4] Uploading {sums_file.name} and {readme_file.name}...")
    upload_file_to_bucket(bucket_url, sums_file, token)
    upload_file_to_bucket(bucket_url, readme_file, token)

    receipt_path = stage_dir.parent / f"{stage_dir.name}.deposit.json"
    receipt = {
        "environment": "production" if args.production else "sandbox",
        "deposition_id": dep_id,
        "reserved_doi": reserved_doi,
        "html_url": html_url,
        "published": False,
    }

    if args.publish:
        print(f"[4/4] Publishing deposition {dep_id}...")
        pub = api_json("POST", f"{base}/deposit/depositions/{dep_id}/actions/publish", token)
        receipt["published"] = True
        receipt["doi"] = pub.get("doi", reserved_doi)
        receipt["record_html"] = pub.get("links", {}).get("record_html", html_url)
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(f"[DONE] Published! DOI: {receipt['doi']} | URL: {receipt['record_html']}")
    else:
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print("[4/4] Kept in PRIVATE DRAFT (safe for double-anonymous review).")
        print(f"[DONE] Review draft in browser: {html_url}")
        print(f"       Reserved DOI (resolves once published): {reserved_doi}")
        prod_flag = "--production " if args.production else ""
        print(
            f"       To publish THIS reserved draft upon acceptance:\n"
            f"       python3 replication/upload_to_zenodo.py {prod_flag}--deposition-id {dep_id} \\\n"
            f"         --publish --yes-i-know-this-is-public-and-permanent --package-dir {stage_dir}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
