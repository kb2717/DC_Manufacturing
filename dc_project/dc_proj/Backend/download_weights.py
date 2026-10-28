"""Put the trained weights in Backend/weights/.

    python download_weights.py

1. Downloads them from the project's GitHub Release (set RELEASE_URL below once
   you have uploaded resnet50_damage.pth and yolo26_damage.pt as release assets).
2. Falls back to extracting them from this repository's git history, where the
   originally trained weights were committed (needs a full, non-shallow clone).
"""
import subprocess
import sys
import urllib.request
from pathlib import Path

BACKEND = Path(__file__).resolve().parent
WEIGHTS = BACKEND / "weights"

RELEASE_URL = "https://github.com/kb2717/DC_Manufacturing/releases/download/v1.0"

# file name -> git blob that holds the original trained weights
FILES = {
    # ResNet-50 classifier (commit 79e0d71, "Resnet50")
    "resnet50_damage.pth": "bacefefe269a9aa78b8595b40dbdfd818ce49949",
    # YOLO26s-seg, 1280px, 150 epochs on VehiDE — best.pt (commit c2667c6)
    "yolo26_damage.pt": "8bd85d8bb347144820097e2733fef5fedfa51451",
}


def from_release(name, dest):
    url = f"{RELEASE_URL}/{name}"
    try:
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as exc:
        print(f"  not available ({exc})")
        dest.unlink(missing_ok=True)
        return False


def from_git_history(blob, dest):
    try:
        with open(dest, "wb") as f:
            subprocess.run(["git", "cat-file", "-p", blob], cwd=BACKEND, stdout=f, check=True)
        return dest.stat().st_size > 1_000_000
    except Exception as exc:
        print(f"  git extraction failed ({exc})")
        dest.unlink(missing_ok=True)
        return False


def main():
    WEIGHTS.mkdir(exist_ok=True)
    ok = True
    for name, blob in FILES.items():
        dest = WEIGHTS / name
        if dest.exists() and dest.stat().st_size > 1_000_000:
            print(f"{name}: already present")
            continue
        if from_release(name, dest) or from_git_history(blob, dest):
            print(f"{name}: saved to {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        else:
            ok = False
            print(f"{name}: FAILED — train it (notebooks/train_vehide.ipynb) or copy it in by hand.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
