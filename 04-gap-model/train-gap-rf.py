#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train the gap-RF models for the single-season rice internal correction.

For each rice type and each of the 7 intervals between adjacent
phenological stages, an RF regression model is trained with the 46 8-day
NDVI composites as features and the interval (gap) in days as target. The
gap-RF models are later used by correct-pheno-raster.py to rebuild the
phenological sequence of inconsistent pixels.

Example:
    python 04-gap-model/train-gap-rf.py
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearnex import patch_sklearn
from sklearn.ensemble import RandomForestRegressor

patch_sklearn()
sys.path.append(str(Path(__file__).resolve().parents[1] / "lib"))
from common import BAND_COLS, PHENOS, add_data_root_arg



CROPS = ["single", "early", "late"]
GAP_NAMES = [
    "transplanting_to_regreening",
    "regreening_to_tillering",
    "tillering_to_stem_elongation",
    "stem_elongation_to_booting",
    "booting_to_heading",
    "heading_to_milk_ripening",
    "milk_ripening_to_maturity",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train the gap-RF models.")
    add_data_root_arg(parser)
    parser.add_argument("--sample-curve-dir", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--min-inner-gap", type=int, default=2)
    parser.add_argument("--crops", nargs="+", default=CROPS, choices=CROPS)
    parser.add_argument("--n-jobs", type=int, default=5)
    return parser.parse_args()


def is_monotonic_increasing(arr, axis=1, min_gap=2):
    return np.all(np.diff(arr, axis=axis) >= min_gap, axis=axis)


def main():
    args = parse_args()
    sample_curve_dir = args.sample_curve_dir or (args.data_root / "sample-curve")
    model_dir = args.model_dir or (args.data_root / "models")
    model_dir.mkdir(parents=True, exist_ok=True)

    for ricetype in args.crops:
        print(f"\n{'=' * 60}")
        print(f"  {ricetype}")
        print(f"{'=' * 60}")

        pattern = f"{sample_curve_dir}/sample-{ricetype}-*-*-NDVI-V2.feather"
        files = sorted(glob.glob(pattern))
        if not files:
            print("  no NDVI files found, skip")
            continue
        print(f"  {len(files)} files")

        chunks = []
        for fp in files:
            df = pd.read_feather(fp)
            # Drop rows with any missing NDVI composite.
            ndvi = df[BAND_COLS]
            valid = ndvi.notna().all(axis=1)
            df = df[valid].reset_index(drop=True)
            if len(df) == 0:
                continue
            # Only strictly increasing phenological sequences are kept.
            ph = df[PHENOS].values.astype(np.float32)
            valid_mask = is_monotonic_increasing(ph, min_gap=args.min_inner_gap)
            df = df[valid_mask].reset_index(drop=True)
            ph = df[PHENOS].values.astype(np.float32)

            gaps = np.diff(ph, axis=1)  # shape (n, 7)
            for gi, gn in enumerate(GAP_NAMES):
                df[gn] = gaps[:, gi]
            chunks.append(df)

        if not chunks:
            print("  no valid training data")
            continue

        full_df = pd.concat(chunks, ignore_index=True)
        print(f"  valid training samples: {len(full_df)}")

        X = full_df[BAND_COLS].values.astype(np.float32)

        for gi, gn in enumerate(GAP_NAMES):
            y = full_df[gn].values.astype(np.float32)
            valid_y = ~np.isnan(y)
            Xi, yi = X[valid_y], y[valid_y]
            print(f"  {gn}: valid samples {len(yi)}")

            if len(yi) < 100:
                print("    too few samples, skip")
                continue

            model = RandomForestRegressor(
                n_estimators=args.n_estimators, n_jobs=args.n_jobs,
            )
            model.fit(Xi, yi)

            outpath = model_dir / f"{ricetype}-{gn}-NDVI46-RF.joblib.z"
            joblib.dump(model, outpath)
            print(f"    model saved: {outpath}")

    print("\ndone!")


if __name__ == "__main__":
    main()
