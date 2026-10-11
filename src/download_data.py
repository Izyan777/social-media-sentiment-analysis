"""Download the public Roman Urdu sentiment dataset (Smat26/Roman-Urdu-Dataset).

Original data compiled by Zareen Sharf (UCI "Roman Urdu Data Set"); licensed GPL-3.0.
We fetch it with `git clone` instead of copying it into this repository.

Usage:  python -m src.download_data
"""
import subprocess
import sys

from src.config import ROMAN_URDU_CSV, ROMAN_URDU_REPO_DIR, ROMAN_URDU_REPO_URL, ensure_dirs


def main() -> None:
    ensure_dirs()
    if ROMAN_URDU_CSV.exists():
        print(f"Already downloaded: {ROMAN_URDU_CSV}")
        return
    print(f"Cloning {ROMAN_URDU_REPO_URL} ...")
    subprocess.run(
        ["git", "clone", "--depth", "1", ROMAN_URDU_REPO_URL, str(ROMAN_URDU_REPO_DIR)],
        check=True,
    )
    if not ROMAN_URDU_CSV.exists():
        sys.exit("Clone finished but the expected CSV is missing - the upstream repo layout may have changed.")
    print(f"Done: {ROMAN_URDU_CSV}")


if __name__ == "__main__":
    main()
