#!/usr/bin/env python3
"""
Download the Modified Swiss Dwellings (MSD) dataset from Kaggle.

Usage:
    python data/download_msd.py --output_dir data/msd_raw

Requires either:
    - kaggle CLI configured with API credentials (~/.kaggle/kaggle.json)
    - Or manual download from https://www.kaggle.com/datasets/caspervanengelenburg/modified-swiss-dwellings
"""

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


KAGGLE_DATASET = "caspervanengelenburg/modified-swiss-dwellings"

EXPECTED_DIRS = [
    "train/structure_in",
    "train/graph_in",
    "train/full_out",
    "train/graph_out",
    "test/structure_in",
    "test/graph_in",
]


def download_with_kaggle(output_dir: str) -> bool:
    """Attempt to download the dataset using the kaggle CLI."""
    if shutil.which("kaggle") is None:
        print("[INFO] kaggle CLI not found on PATH.")
        return False

    # Check for credentials
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    kaggle_env = os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    if not kaggle_json.exists() and not kaggle_env:
        print("[INFO] Kaggle credentials not found (~/.kaggle/kaggle.json or env vars).")
        return False

    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Downloading dataset '{KAGGLE_DATASET}' to {output_dir} ...")
    try:
        subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", KAGGLE_DATASET,
                "-p", output_dir,
            ],
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] kaggle download failed: {e}")
        return False


def extract_zip(output_dir: str) -> bool:
    """Find and extract the downloaded zip file."""
    zip_candidates = list(Path(output_dir).glob("*.zip"))
    if not zip_candidates:
        print("[ERROR] No zip file found in", output_dir)
        return False

    zip_path = zip_candidates[0]
    print(f"[INFO] Extracting {zip_path} ...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(output_dir)
        print(f"[INFO] Extraction complete. Removing {zip_path}")
        zip_path.unlink()
        return True
    except zipfile.BadZipFile as e:
        print(f"[ERROR] Bad zip file: {e}")
        return False


def verify_structure(output_dir: str) -> bool:
    """Verify the expected directory structure exists after extraction."""
    base = Path(output_dir)
    all_ok = True

    # The dataset may be nested one level deeper (e.g., msd_raw/modified-swiss-dwellings/train/...)
    # Try to detect and handle that.
    possible_roots = [base]
    for child in base.iterdir():
        if child.is_dir() and child.name not in ("train", "test"):
            possible_roots.append(child)

    root = None
    for candidate in possible_roots:
        if (candidate / "train").is_dir():
            root = candidate
            break

    if root is None:
        print("[ERROR] Could not find 'train/' directory inside", output_dir)
        print("[HINT] Contents:", [p.name for p in base.iterdir()])
        return False

    # If the actual root is a subdirectory, move contents up
    if root != base:
        print(f"[INFO] Dataset found in subdirectory '{root.name}', relocating to {output_dir}")
        for item in root.iterdir():
            dest = base / item.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            shutil.move(str(item), str(dest))
        root.rmdir()

    for expected in EXPECTED_DIRS:
        path = base / expected
        if path.is_dir():
            n_files = len(list(path.iterdir()))
            print(f"  [OK] {expected}/ ({n_files} files)")
        else:
            print(f"  [MISSING] {expected}/")
            all_ok = False

    return all_ok


def print_manual_instructions(output_dir: str):
    """Print instructions for manual download."""
    print("\n" + "=" * 70)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 70)
    print(f"""
1. Go to: https://www.kaggle.com/datasets/{KAGGLE_DATASET}
2. Click 'Download' (you need a free Kaggle account)
3. Save the zip file to: {output_dir}/
4. Extract it so the structure is:
   {output_dir}/
   ├── train/
   │   ├── structure_in/
   │   ├── graph_in/
   │   ├── full_out/
   │   └── graph_out/
   └── test/
       ├── structure_in/
       ├── graph_in/
       └── ...

Alternatively, set up kaggle CLI:
    pip install kaggle
    # Create API token at https://www.kaggle.com/settings
    # Save to ~/.kaggle/kaggle.json
    python {__file__} --output_dir {output_dir}
""")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Download the Modified Swiss Dwellings (MSD) dataset from Kaggle."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/msd_raw",
        help="Directory to download and extract the dataset into (default: data/msd_raw)",
    )
    parser.add_argument(
        "--verify_only",
        action="store_true",
        help="Only verify the existing directory structure, do not download.",
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)

    if args.verify_only:
        print(f"[INFO] Verifying dataset structure in {output_dir}")
        if verify_structure(output_dir):
            print("\n[SUCCESS] Dataset structure verified.")
        else:
            print("\n[FAIL] Dataset structure incomplete.")
            sys.exit(1)
        return

    # Check if already extracted
    if (Path(output_dir) / "train" / "full_out").is_dir():
        print(f"[INFO] Dataset already present at {output_dir}")
        verify_structure(output_dir)
        return

    # Try kaggle CLI download
    downloaded = download_with_kaggle(output_dir)

    if not downloaded:
        # Check if zip was manually placed
        zips = list(Path(output_dir).glob("*.zip"))
        if zips:
            print(f"[INFO] Found existing zip file: {zips[0]}")
            downloaded = True
        else:
            print_manual_instructions(output_dir)
            sys.exit(1)

    if downloaded:
        if not extract_zip(output_dir):
            print("[ERROR] Extraction failed.")
            sys.exit(1)

        if verify_structure(output_dir):
            print("\n[SUCCESS] MSD dataset downloaded and verified.")
        else:
            print("\n[WARNING] Download complete but some expected directories are missing.")
            print("The dataset structure may differ from expected. Please check manually.")


if __name__ == "__main__":
    main()
