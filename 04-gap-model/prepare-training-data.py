#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare the training data of the gap-RF models.

For each rice type:
  1. Load the sampled phenological dates (merged by merge-sample.py).
  2. Drop samples whose 8 phenological stages are not strictly increasing
     (adjacent interval < ``--min-inner-gap`` days is considered wrong).
  3. Extract the 46 8-day NDVI composites of the sampled pixels from the
     NDVI product, and write one feather file per (province, year).

Example:
    python 04-gap-model/prepare-training-data.py --version 5.4
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
    BAND_COLS, JDAYS, PHENOS, add_data_root_arg, ndvi_fname,
)

CROPS = ["single", "early", "late"]
# The NDVI product organizes the files by province names that contain a space.
PROVINCE_MAP = {"Inner_Mongolia": "Inner Mongolia"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare the gap-RF training data from sampled phenological dates."
    )
    add_data_root_arg(parser)
    parser.add_argument("--sample-dir", type=Path, default=None)
    parser.add_argument("--sample-curve-dir", type=Path, default=None)
    parser.add_argument("--ndvi-dir", type=Path, default=None)
    parser.add_argument("--version", default="5.4")
    parser.add_argument("--n-sample", type=int, default=100_000,
                        help="number of samples drawn (with equal probability) "
                             "from all valid samples.")
    parser.add_argument("--min-inner-gap", type=int, default=2,
                        help="minimum allowed interval (days) between two "
                             "adjacent phenological stages.")
    parser.add_argument("--crops", nargs="+", default=CROPS, choices=CROPS)
    parser.add_argument("--years-start", type=int, default=2001)
    parser.add_argument("--years-end", type=int, default=2024)
    parser.add_argument("--skip-half-province-years", action="store_true", default=True,
                        help="process only about half of the (province, year) "
                             "groups to reduce the workload; disable with "
                             "--no-skip-half-province-years to process all of them.")
    parser.add_argument("--no-skip-half-province-years", dest="skip_half_province_years",
                        action="store_false")
    return parser.parse_args()


def is_monotonic_increasing(arr, axis=0, min_gap=2):
    return np.all(np.diff(arr, axis=axis) >= min_gap, axis=axis)


def main():
    args = parse_args()
    sample_dir = args.sample_dir or (args.data_root / "sample")
    sample_curve_dir = args.sample_curve_dir or (args.data_root / "sample-curve")
    ndvi_dir = args.ndvi_dir or (args.data_root / "ndvi")
    sample_curve_dir.mkdir(parents=True, exist_ok=True)

    years = range(args.years_start, args.years_end + 1)

    for ricetype in args.crops:
        print(f"\n===== {ricetype} =====")
        df = pd.read_feather(
            sample_dir / f"sample-{ricetype}-phenology-all-v{args.version}.feather"
        )
        print(f"  total samples: {len(df)}")

        # 1. Keep only the samples with strictly increasing phenological dates.
        pheno_arr = df[PHENOS].values
        valid_mask = is_monotonic_increasing(pheno_arr, min_gap=args.min_inner_gap)
        df_valid = df[valid_mask].reset_index(drop=True)
        print(f"  valid samples (monotonic): {len(df_valid)}")
        print(f"  removed: {len(df) - len(df_valid)}")

        # 2. Draw a subsample for NDVI extraction.
        df_valid["province"] = df_valid["province"].replace(PROVINCE_MAP)
        if len(df_valid) > args.n_sample:
            df_sample = df_valid.loc[
                np.random.choice(df_valid.index, size=args.n_sample, replace=False)
            ]
        else:
            df_sample = df_valid

        for (province, yr), grp in df_sample.groupby(["province", "year"]):
            if args.skip_half_province_years and (yr + sum(ord(c) for c in province)) % 2 == 0:
                continue
            outfile = sample_curve_dir / f"sample-{ricetype}-{province}-{yr}-NDVI-V2.feather"
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
                        # Vectorized point extraction.
                        lons = xr.DataArray(grp["longitude"].values, dims="z")
                        lats = xr.DataArray(grp["latitude"].values, dims="z")
                        vals = f.sel(x=lons, y=lats, method="nearest").values
                        ndvis[:, j] = vals
                except Exception as e:
                    print(f"      failed to read: {tif_path}  {e}")
                    continue

            # NDVI composites filled with 0 are treated as missing.
            ndvis[ndvis == 0] = np.nan

            temp = pd.DataFrame(ndvis, columns=BAND_COLS, dtype=np.float32)
            out_df = pd.concat([grp, temp], axis=1)
            out_df.to_feather(outfile)
            print(f"    {province}, {yr}: {n_smp} samples -> {outfile}")

    print("\ndone!")


if __name__ == "__main__":
    main()
