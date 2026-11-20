"""Split a class-per-folder image dataset into train / val / test.

The split is stratified: each class is shuffled and cut separately, so every
split keeps the same class balance. Files are copied, the source is untouched.

Example:
    python scripts/split_dataset.py --src data/raw/plantvillage --dst data/processed
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from plantpulse.config import CLASSES, SEED, SPLIT  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def split_files(files, ratios, rng):
    files = sorted(files)
    rng.shuffle(files)
    n = len(files)
    n_train = round(n * ratios["train"])
    n_val = round(n * ratios["val"])
    return {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", type=Path, required=True, help="folder with one subfolder per class")
    parser.add_argument("--dst", type=Path, required=True, help="output folder")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--all-classes", action="store_true",
                        help="use every subfolder in --src instead of config.CLASSES")
    args = parser.parse_args()

    if args.all_classes:
        classes = sorted(p.name for p in args.src.iterdir() if p.is_dir())
    else:
        classes = CLASSES
        missing = [c for c in classes if not (args.src / c).is_dir()]
        if missing:
            sys.exit(f"Missing class folders in {args.src}: {missing}")

    if args.dst.exists() and any(args.dst.iterdir()):
        sys.exit(f"{args.dst} is not empty; remove it first so old files do not mix in.")

    rng = random.Random(args.seed)
    print(f"{'class':50} {'train':>6} {'val':>6} {'test':>6}")
    for cls in classes:
        files = [p for p in (args.src / cls).iterdir() if p.suffix.lower() in IMAGE_EXTS]
        parts = split_files(files, SPLIT, rng)
        for split, items in parts.items():
            out = args.dst / split / cls
            out.mkdir(parents=True, exist_ok=True)
            for f in items:
                shutil.copy2(f, out / f.name)
        print(f"{cls:50} {len(parts['train']):6} {len(parts['val']):6} {len(parts['test']):6}")


if __name__ == "__main__":
    main()
