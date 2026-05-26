#!/usr/bin/env python3
"""
TotalSegmentator v2 zero-shot inference on CHD test set (imagesTs).

Runs the heartchambers_highres task and remaps TS structure names to CHD label IDs.

Usage:
    python models/totalsegmentator/run_zeroshot.py \
        --input-dir data/imagesTs \
        --output-dir results/totalsegmentator/zeroshot \
        [--fast]

Environment: totalseg_env
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np


# TotalSegmentator structure names → CHD label IDs
TS_TO_CHD: dict[str, int] = {
    "heart_ventricle_left":  1,
    "heart_ventricle_right": 2,
    "heart_atrium_left":     3,
    "heart_atrium_right":    4,
    "heart_myocardium":      5,
    "aorta":                 6,
    "pulmonary_artery":      7,
}


def run_case(img_path: Path, out_path: Path, fast: bool) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        ts_out = Path(tmpdir) / "segs"
        ts_out.mkdir()

        cmd = [
            "TotalSegmentator",
            "-i", str(img_path),
            "-o", str(ts_out),
            "--task", "heartchambers_highres",
        ]
        if fast:
            cmd.append("--fast")
        subprocess.run(cmd, check=True)

        nii = nib.load(str(img_path))
        merged = np.zeros(nii.shape[:3], dtype=np.uint8)

        for ts_name, chd_id in TS_TO_CHD.items():
            seg_path = ts_out / f"{ts_name}.nii.gz"
            if seg_path.exists():
                seg = np.asarray(nib.load(str(seg_path)).dataobj, dtype=np.uint8)
                merged[seg > 0] = chd_id
            else:
                print(f"    [missing] {ts_name}")

        nib.save(nib.Nifti1Image(merged, nii.affine, nii.header), str(out_path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/imagesTs"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/totalsegmentator/zeroshot"))
    parser.add_argument("--fast", action="store_true", help="3mm resolution (quicker sanity check)")
    parser.add_argument("--cases", nargs="*", help="Specific case IDs (default: all)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(args.input_dir.glob("*_0000.nii.gz"))
    if not images:
        images = sorted(args.input_dir.glob("*.nii.gz"))
    if args.cases:
        images = [p for p in images if any(c in p.name for c in args.cases)]

    print(f"TotalSegmentator zero-shot: {len(images)} cases  fast={args.fast}")

    failed = 0
    for img_path in images:
        case_id = img_path.name.replace("_0000.nii.gz", "").replace(".nii.gz", "")
        out_path = args.output_dir / f"{case_id}.nii.gz"
        if out_path.exists():
            print(f"  [skip] {case_id}")
            continue
        print(f"  {case_id}...", end=" ", flush=True)
        try:
            run_case(img_path, out_path, args.fast)
            print("done")
        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1

    print(f"\nPredictions written to {args.output_dir}")
    if failed:
        print(f"ERROR: {failed}/{len(images)} cases failed", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
