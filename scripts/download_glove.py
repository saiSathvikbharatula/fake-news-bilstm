"""Download GloVe 6B embeddings and extract the 100-dimensional file.

Usage (from the project root):
    python scripts/download_glove.py

The zip is ~820 MB. If the download is slow, grab "glove.6B.100d.txt" from Kaggle
(search "glove6b100dtxt") and put it at glove/glove.6B.100d.txt instead.
"""
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://nlp.stanford.edu/data/glove.6B.zip"
ROOT = Path(__file__).resolve().parent.parent
GLOVE_DIR = ROOT / "glove"
TARGET = "glove.6B.100d.txt"


def _progress(block, block_size, total):
    done = block * block_size
    if total > 0:
        pct = min(100, done * 100 / total)
        sys.stdout.write(f"\rDownloading ... {pct:5.1f}% ({done / 1e6:,.0f} MB)")
        sys.stdout.flush()


def main():
    GLOVE_DIR.mkdir(exist_ok=True)
    out = GLOVE_DIR / TARGET
    if out.exists():
        print(f"{out} already exists - nothing to do.")
        return
    zip_path = GLOVE_DIR / "glove.6B.zip"
    if not zip_path.exists():
        urllib.request.urlretrieve(URL, zip_path, _progress)
        print()
    print(f"Extracting {TARGET} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extract(TARGET, GLOVE_DIR)
    zip_path.unlink()
    print(f"Done -> {out}")


if __name__ == "__main__":
    main()
