#!/usr/bin/env python3
"""
convert_npy_to_kitti_bin.py
Convert a folder of .npy point clouds to KITTI-style .bin (float32 x,y,z[,intensity]).

Usage:
  python convert_npy_to_kitti_bin.py /path/to/in_dir /path/to/out_dir [--cols 3|4] [--recursive] [--overwrite]

Notes:
- Each .npy is expected to be shape (N, M) with at least 3 columns (x,y,z). If M>=4 and --cols is 4,
  the 4th column (intensity) will be included. Extra columns are ignored.
- Output .bin is a raw float32 binary with shape (N, C) flattened, compatible with many LiDAR pipelines.
"""
import argparse
import sys
from pathlib import Path
import numpy as np

def convert_file(src: Path, dst: Path, cols: int, overwrite: bool):
    if dst.exists() and not overwrite:
        print(f"[skip exists] {dst}")
        return True
    try:
        arr = np.load(src, allow_pickle=False)
    except Exception as e:
        print(f"[error load] {src}: {e}", file=sys.stderr)
        return False

    if arr.ndim != 2 or arr.shape[1] < 3:
        print(f"[bad shape] {src}: expected (N, M>=3), got {arr.shape}", file=sys.stderr)
        return False

    use_cols = cols if cols in (3,4) else (4 if arr.shape[1] >= 4 else 3)
    if use_cols == 4 and arr.shape[1] < 4:
        print(f"[fallback] {src}: requested 4 cols but only {arr.shape[1]} present -> using 3")
        use_cols = 3

    out = arr[:, :use_cols].astype(np.float32, copy=False)
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.tofile(dst)
    print(f"[ok] {src.name} -> {dst.name}  (points={out.shape[0]}, cols={use_cols})")
    return True

def main():
    ap = argparse.ArgumentParser(description="Convert .npy point clouds to KITTI-style .bin")
    ap.add_argument("in_dir", type=Path, help="Input directory containing .npy files")
    ap.add_argument("out_dir", type=Path, help="Output directory for .bin files")
    ap.add_argument("--cols", type=int, choices=[3,4], default=None,
                    help="Number of columns to write (default: 4 if available else 3)")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing .bin files")
    args = ap.parse_args()

    pattern = "**/*.npy" if args.recursive else "*.npy"
    npy_files = sorted(args.in_dir.glob(pattern))
    if not npy_files:
        print(f"No .npy files found in {args.in_dir} (recursive={args.recursive})")
        sys.exit(1)

    ok = 0
    for src in npy_files:
        rel = src.relative_to(args.in_dir)
        dst = (args.out_dir / rel).with_suffix(".bin")
        if convert_file(src, dst, args.cols, args.overwrite):
            ok += 1

    print(f"Done. Converted {ok}/{len(npy_files)} files. Output at: {args.out_dir.resolve()}")

if __name__ == "__main__":
    main()
