#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare the training data of the early-late gap-RF model.

  1. Load the early-late samples (merged by merge-sample-early-late.py).
  2. Drop samples whose interval between early maturity and late
     transplanting is smaller than ``--min-gap`` days.
  3. Extract the 46 8-day NDVI composites of the sampled pixels and write
     one feather file per (province, year).

Example:
    python 04-gap-model/prepare-early-late-training-data.py --version 5-unified2-el-filled
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from rioxarray import open_rasterio

sys.path.append(str(Path(__file__).resolve().parents[1] / "lib"))
from common import (
    BAND_COLS, JDAYS, add_data_root_arg, ndvi_fname,
)

PROVINCE_MAP = {"Inner_Mongolia": "Inner Mongolia"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare the early-late gap-RF training data."
    )
    add_data_root_arg(parser)
    parser.add_argument("--sample-dir", type=Path, default=None)
    parser.add_argument("--sample-curve-dir", type=Path, default=None)
    parser.add_argument("--ndvi-dir", type=Path, default=None)
    parser.add_argument("--version", default="5-unified2-el-filled")
    parser.add_argument("--min-gap", type=int, default=1,
                        help="minimum allowed interval (days) between late "
                             "transplanting and early maturity.")
    parser.add_argument("--n-sample", type=int, default=100_000)
    parser.add_argument("--years-start", type=int, default=2001)
    parser.add_argument("--years-end", type=int, default=2024)
    return parser.parse_args()


def main():
    args = parse_args()
    sample_dir = args.sample_dir or (args.data_root / "sample")
    sample_curve_dir = args.sample_curve_dir or (args.data_root / "sample-curve")
    ndvi_dir = args.ndvi_dir or (args.data_root / "ndvi")
    sample_curve_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_feather(
        sample_dir / f"sample-early-late-phenology-all-v{args.version}.feather"
    )
    print(f"  total samples: {len(df)}")

    # Drop samples with an invalid interval.
    df_valid = df[
        (df["gap"] >= args.min_gap) &
        df["early_maturity"].notna() &
        df["late_transplanting"].notna()
    ].reset_index(drop=True)
    print(f"  valid samples (gap >= {args.min_gap}): {len(df_valid)}")
    print(f"  removed: {len(df) - len(df_valid)}")

    df_valid["province"] = df_valid["province"].replace(PROVINCE_MAP)
    if len(df_valid) > args.n_sample:
        df_sample = df_valid.loc[
            np.random.choice(df_valid.index, size=args.n_sample, replace=False)
        ]
    else:
        df_sample = df_valid

    for (province, yr), grp in df_sample.groupby(["province", "year"]):
        outfile = sample_curve_dir / f"sample-early-late-{province}-{yr}-NDVI-V2.feather"
        if os.path.isfile(outfile):
            print(f"    exists: {outfile}")
            continue

        grp = grp.reset_index(drop=True)
        n_smp = len(grp)
        ndvis = np.ones([n_smp, len(JDAYS)], dtype=np.float32) * np.nan

        for j, jday in enumerate(JDAYS):
            tif_path = ndvi_dir / province / ndvi_fname(province, yr, jday)
            if not os.path.isfile(tif_path):
                continue
            try:
                with open_rasterio(tif_path) as f:
                    lons = xr.DataArray(grp["longitude"].values, dims="z")
                    lats = xr.DataArray(grp["latitude"].values, dims="z")
                    vals = f.sel(x=lons, y=lats, method="nearest").values
                    ndvis[:, j] = vals
            except Exception as e:
                print(f"      failed to read: {tif_path}  {e}")
                continue

        ndvis[ndvis == 0] = np.nan

        temp = pd.DataFrame(ndvis, columns=BAND_COLS, dtype=np.float32)
        out_df = pd.concat([grp, temp], axis=1)
        out_df.to_feather(outfile)
        print(f"    {province}, {yr}: {n_smp} samples -> {outfile}")

    print("\ndone!")


if __name__ == "__main__":
    main()
