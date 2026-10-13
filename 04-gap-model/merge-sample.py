#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge the per-province-year sample CSVs (outputs of pick-sample.jl) into one
table per rice type. Up to ``--n-per-year`` samples are kept for each year.

Outputs:
    <sample-dir>/sample-<crop>-phenology-all-v<version>.feather
    <sample-dir>/sample-<crop>-phenology-all-v<version>.parquet   (with geometry)

Example:
    python 04-gap-model/merge-sample.py --version 5.4
"""
import argparse
import glob
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "lib"))
from common import add_data_root_arg

CROPS = ["single", "early", "late"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge per-province-year sample files into one table per rice type."
    )
    add_data_root_arg(parser)
    parser.add_argument("--sample-dir", type=Path, default=None)
    parser.add_argument("--version", default="5.4",
                        help="version tag of the sample files (must match "
                             "--version-out of pick-sample.jl).")
    parser.add_argument("--n-per-year", type=int, default=5000)
    parser.add_argument("--crops", nargs="+", default=CROPS,
                        choices=CROPS, help="rice types to merge.")
    parser.add_argument("--years-start", type=int, default=2001)
    parser.add_argument("--years-end", type=int, default=2024)
    return parser.parse_args()


def latlon_to_wkt(lon, lat):
    return f"POINT ({lon} {lat})"


def sheet_to_gsheet(sheet):
    wkt = pd.Series(np.vectorize(latlon_to_wkt)(
        sheet["longitude"].to_list(), sheet["latitude"].to_list()
    ))
    gs = gpd.GeoSeries.from_wkt(wkt)
    return gpd.GeoDataFrame(sheet, geometry=gs, crs="EPSG:4326")


def main():
    args = parse_args()
    sample_dir = args.sample_dir or (args.data_root / "sample")
    province_dir = sample_dir / "province"
    version = args.version

    for crop in args.crops:
        sheet0 = pd.DataFrame()
        for yr in range(args.years_start, args.years_end + 1):
            flist = glob.glob(
                f"{province_dir}/sample-*-{yr}-{crop}-v{version}.csv"
            )
            if not flist:
                continue
            sheet = pd.concat([pd.read_csv(file) for file in flist], ignore_index=True)
            sheet["year"] = yr
            if len(sheet) <= args.n_per_year:
                index = np.random.choice(sheet.index, size=args.n_per_year, replace=True)
            else:
                index = np.random.choice(sheet.index, size=args.n_per_year, replace=False)
            sheet1 = sheet.iloc[index]
            sheet0 = pd.concat([sheet0, sheet1], ignore_index=True)

        outfile = sample_dir / f"sample-{crop}-phenology-all-v{version}.feather"
        sheet0.to_feather(outfile)
        print(f"saved: {outfile}")

        gsheet0 = sheet_to_gsheet(sheet0)
        gsheet0.to_parquet(
            sample_dir / f"sample-{crop}-phenology-all-v{version}.parquet",
            index=False,
        )


if __name__ == "__main__":
    main()
