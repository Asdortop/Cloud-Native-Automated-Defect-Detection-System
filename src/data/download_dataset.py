"""
download_dataset.py
-------------------
Downloads the NEU Metal Surface Defect Dataset from Kaggle and extracts it
into <project_root>/data/raw.

Prerequisites
-------------
* kaggle CLI installed  : pip install kaggle
* Credentials via env   : KAGGLE_USERNAME and KAGGLE_KEY must be set.

Usage
-----
    python src/data/download_dataset.py [--data-dir data/raw] [--keep-zip]
"""

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Dataset identifiers
# ---------------------------------------------------------------------------
# Kaggle dataset slug: <owner>/<dataset-name>
KAGGLE_DATASET = "kaustubhdikshit/neu-surface-defect-database"

# Expected zip filename produced by the Kaggle CLI
DATASET_ZIP = "neu-surface-defect-database.zip"


def check_credentials() -> None:
    """Validate that Kaggle API credentials are present in the environment."""
    missing = [v for v in ("KAGGLE_USERNAME", "KAGGLE_KEY") if not os.getenv(v)]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variable(s): {', '.join(missing)}.\n"
            "Set KAGGLE_USERNAME and KAGGLE_KEY before running this script."
        )


def download_dataset(output_dir: Path) -> Path:
    """
    Invoke the Kaggle CLI to download the NEU dataset zip into *output_dir*.

    Parameters
    ----------
    output_dir : Path
        Destination directory for the downloaded zip file.

    Returns
    -------
    Path
        Full path to the downloaded zip archive.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / DATASET_ZIP

    if zip_path.exists():
        print(f"[INFO] Archive already exists at '{zip_path}'. Skipping download.")
        return zip_path

    print(f"[INFO] Downloading dataset '{KAGGLE_DATASET}' ...")
    cmd = [
        sys.executable, "-m", "kaggle",
        "datasets", "download",
        "--dataset", KAGGLE_DATASET,
        "--path", str(output_dir),
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Kaggle CLI download failed with exit code {exc.returncode}."
        ) from exc

    if not zip_path.exists():
        # The CLI may name the file differently; try to find any zip in the dir.
        zips = list(output_dir.glob("*.zip"))
        if not zips:
            raise FileNotFoundError(
                f"Download appeared to succeed but no .zip file found in '{output_dir}'."
            )
        zip_path = zips[0]
        print(f"[INFO] Found archive: '{zip_path}'")

    print(f"[INFO] Download complete: '{zip_path}'")
    return zip_path


def extract_dataset(zip_path: Path, extract_to: Path) -> None:
    """
    Extract *zip_path* into *extract_to*, skipping files that already exist.

    Parameters
    ----------
    zip_path   : Path  - Path to the downloaded zip archive.
    extract_to : Path  - Directory where contents will be extracted.
    """
    extract_to.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Extracting '{zip_path.name}' -> '{extract_to}' ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        total = len(members)
        for idx, member in enumerate(members, start=1):
            dest = extract_to / member
            if dest.exists():
                continue  # skip already-extracted files
            zf.extract(member, path=extract_to)
            if idx % 500 == 0 or idx == total:
                print(f"[INFO]   {idx}/{total} files extracted ...")

    print(f"[INFO] Extraction complete. Dataset available at '{extract_to}'.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and extract the NEU Surface Defect Dataset from Kaggle."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "raw",
        help="Destination directory for raw dataset files (default: <root>/data/raw).",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        default=False,
        help="Keep the downloaded zip archive after extraction.",
    )
    args = parser.parse_args()

    raw_dir: Path = args.data_dir.resolve()

    # 1. Validate credentials
    check_credentials()

    # 2. Download
    zip_path = download_dataset(raw_dir)

    # 3. Extract
    extract_dataset(zip_path, raw_dir)

    # 4. Optionally remove the zip to save disk space
    if not args.keep_zip and zip_path.exists():
        zip_path.unlink()
        print(f"[INFO] Removed archive '{zip_path}'.")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
