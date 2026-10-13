#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge the per-province-year early-late sample CSVs (outputs of
pick-sample-early-late.jl) into one table.

Outputs:
    <sample-dir>/sample-early-late-phenology-all-v<version>.feather
    <sample-dir>/sample-early-late-phenology-all-v<version>.parquet  (with geometry)

Example:
    python 04-gap-model/merge-sample-early-late.py --version 5-unified2-el-filled
"""
import argparse
import os
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "lib"))
from common import PLACES_DOUBLE, add_data_root_arg


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge the early-late sample files into one table."
    )
    add_data_root_arg(parser)
    parser.add_argument("--sample-dir", type=Path, default=None)
    parser.add_argument("--version", default="5-unified2-el-filled")
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
    inpath = sample_dir / "early-late"
    version = args.version

    outfile = sample_dir / f"sample-early-late-phenology-all-v{version}.feather"
    if not os.path.isfile(outfile):
        sheet0 = pd.DataFrame()
        for place in PLACES_DOUBLE:
            for yr in range(args.years_start, args.years_end + 1):
                file = inpath / f"sample-{place}-{yr}-v{version}.csv"
                if not os.path.isfile(file):
                    continue
                sheet = pd.read_csv(file)
                sheet["province"] = place
                sheet["year"] = yr
                sheet0 = pd.concat([sheet0, sheet], ignore_index=True)
        sheet0.to_feather(outfile)
        print(f"saved: {outfile}")

    outfile2 = sample_dir / f"sample-early-late-phenology-all-v{version}.parquet"
    if not os.path.isfile(outfile2):
        sheet0 = pd.read_feather(outfile)
        gsheet0 = sheet_to_gsheet(sheet0)
        gsheet0.to_parquet(outfile2, index=False)
        print(f"saved: {outfile2}")


if __name__ == "__main__":
    main()
