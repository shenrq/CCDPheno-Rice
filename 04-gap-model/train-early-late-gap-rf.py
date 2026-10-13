#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train the early-late gap-RF model.

The target is the interval (gap) between late-season transplanting and
early-season maturity (``gap = late_transplanting - early_maturity``), which
is predicted from the 46 8-day NDVI composites. The model is used by
correct-early-late-raster.py to fix the double-season rice rotation interval.

Example:
    python 04-gap-model/train-early-late-gap-rf.py
"""
import argparse
import glob
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearnex import patch_sklearn
from sklearn.ensemble import RandomForestRegressor

patch_sklearn()
sys.path.append(str(Path(__file__).resolve().parents[1] / "lib"))
from common import BAND_COLS, add_data_root_arg


def parse_args():
    parser = argparse.ArgumentParser(description="Train the early-late gap-RF model.")
    add_data_root_arg(parser)
    parser.add_argument("--sample-curve-dir", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    sample_curve_dir = args.sample_curve_dir or (args.data_root / "sample-curve")
    model_dir = args.model_dir or (args.data_root / "models")
    model_dir.mkdir(parents=True, exist_ok=True)

    pattern = f"{sample_curve_dir}/sample-early-late-*-*-NDVI-V2.feather"
    files = sorted(glob.glob(pattern))
    print(f"{len(files)} files")

    chunks = []
    for fp in files:
        df = pd.read_feather(fp)
        ndvi = df[BAND_COLS]
        valid = ndvi.notna().all(axis=1)
        df = df[valid].reset_index(drop=True)
        if len(df) == 0:
            continue
        chunks.append(df)

    if not chunks:
        print("no valid training data")
        raise SystemExit

    full_df = pd.concat(chunks, ignore_index=True)
    print(f"valid training samples: {len(full_df)}")

    # Target: interval between early maturity and late transplanting.
    y = full_df["gap"].values.astype(np.float32)
    valid_y = ~np.isnan(y)
    X = full_df[BAND_COLS].values.astype(np.float32)
    X, y = X[valid_y], y[valid_y]
    print(f"gap valid samples: {len(y)}")

    model = RandomForestRegressor(
        n_estimators=args.n_estimators, n_jobs=args.n_jobs,
    )
    model.fit(X, y)

    outpath = model_dir / "early-late-gap-NDVI46-RF.joblib.z"
    joblib.dump(model, outpath)
    print(f"model saved: {outpath}")


if __name__ == "__main__":
    main()
