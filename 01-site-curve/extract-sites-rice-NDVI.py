#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract NDVI time series of rice pixels around AMS (Agricultural Meteorological Stations) from the 30 m 8-day NDVI product.

For each station-year record in the observation table, the NDVI time series
of all rice pixels within a buffer around the station are collected and
saved as a .mat file (46 x N array, N = number of rice pixels).

This is Step 1 of the pipeline. Run it once for each rice type
(``--crop single|early|late``).

Example:
    python 01-site-curve/extract-sites-rice-NDVI.py --crop single
"""
import argparse
import gc
import glob
import os
import re
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import scipy.io as sio

sys.path.append(str(Path(__file__).resolve().parents[1] / "lib"))
from common import (  # noqa: E402
    RES30, add_data_root_arg,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract NDVI time series of rice pixels around AMS sites."
    )
    add_data_root_arg(parser)
    parser.add_argument(
        "--crop", required=True, choices=["single", "early", "late"],
        help="rice type: single-season, double-season early, or double-season late.",
    )
    parser.add_argument("--obs-dir", type=Path, default=None,
                        help="directory of the station-observation GPKG files.")
    parser.add_argument("--obs-file", type=Path, default=None,
                        help="explicit path of the observation GPKG "
                             "(overrides --obs-dir).")
    parser.add_argument("--years-start", type=int, default=2000)
    parser.add_argument("--years-end", type=int, default=2015)
    parser.add_argument("--buffer", type=float, default=0.05,
                        help="buffer size in degrees around each station.")
    parser.add_argument("--rice-map-version", type=int, default=1,
                        help="version number in the rice-map file names.")
    parser.add_argument("--ndvi-dir", type=Path, default=None,
                        help="directory of the 8-day NDVI product, organized "
                             "into one subdirectory per province.")
    parser.add_argument("--rice-dir", type=Path, default=None,
                        help="directory of the rice distribution maps, "
                             "organized into one subdirectory per year.")
    parser.add_argument("--sites-dir", type=Path, default=None,
                        help="output directory of the extracted NDVI curves "
                             "(one subdirectory per rice type is created).")
    return parser.parse_args()


def main():
    args = parse_args()
    crop = args.crop

    # Default data directories (derived from --data-root).
    obs_dir = args.obs_dir or (args.data_root / "observation")
    ndvi_dir = args.ndvi_dir or (args.data_root / "ndvi")
    rice_dir = args.rice_dir or (args.data_root / "rice-map")
    sites_dir = args.sites_dir or (args.data_root / "sites-curve")

    # Observation table of the given rice type (GPKG with columns:
    # station, lat, lon, altitude, PAC, province, year, and one column per
    # phenological stage).
    obs_file = args.obs_file or (obs_dir / f"{crop}-rice-phenology.xlsx")

    obs = pd.read_excel(obs_file)
    # | station | lat  | lon    | altitude | PAC    | province     | year | transplanting | ... | maturity |
    # | ------- | ---- | ------ | -------- | ------ | ------------ | ---- | ------------- | ... | -------- |
    # | XXXXX   | XX.X | 1XX.XX | XX.X     | 230000 | Heilongjiang | 2001 | 142           | ... | 259      |
    # ...
    sites = obs["station"].drop_duplicates().to_list()

    ricecode = 2 if crop in ("early", "late") else 1
    outpath = sites_dir / crop
    outpath.mkdir(parents=True, exist_ok=True)
    vmap = args.rice_map_version
    buffer = args.buffer

    for site in sites:
        for yr in range(args.years_start, args.years_end + 1):
            row = obs[(obs["station"] == site) & (obs["year"] == yr)]
            if len(row) == 0:
                continue
            province = re.sub(r" ", "_", row["province"].to_list()[0])
            # Yunnan has no double-season rice.
            if province == "Yunnan" and crop != "single":
                continue
            # Fused rice distribution map of the corresponding year.
            cropfile = rice_dir / str(yr) / f"CCD-Rice-{province}-{yr}-v{vmap}.tif"
            if not os.path.isfile(cropfile):
                continue
            lat = row["lat"].to_list()[0]
            lon = row["lon"].to_list()[0]
            outfile = outpath / f"{site}-{yr}-NDVI-time_series-v{vmap}-buffer-{buffer}.mat"
            if os.path.isfile(outfile):
                continue

            print(site, province, yr)
            # All 8-day NDVI composites of the year (file names are matched by
            # the province name and year).
            flist = sorted(glob.glob(f"{ndvi_dir}/{province}/{province}*_{yr}*.tif"))
            if len(flist) == 0:
                continue
            xsel = slice(lon - buffer, lon + buffer)
            ysel = slice(lat + buffer, lat - buffer)

            # Read the NDVI stack (n_time, h, w) within the buffer.
            from rioxarray import open_rasterio
            x_ndvi = open_rasterio(flist[0]).sel(x=xsel, y=ysel).x.values
            y_ndvi = open_rasterio(flist[0]).sel(x=xsel, y=ysel).y.values
            ndvi = np.array([
                open_rasterio(file).sel(x=xsel, y=ysel).values[0, :, :]
                for file in flist
            ])

            # Rice map within the same buffer (pixels of the target rice type).
            ricemap = open_rasterio(cropfile).sel(x=xsel, y=ysel)
            x_map = ricemap.x.values
            y_map = ricemap.y.values
            ricemap = ricemap.values == ricecode

            # Intersection of the two grids.
            left = np.maximum(x_ndvi[0], x_map[0])
            top = np.minimum(y_ndvi[0], y_map[0])
            right = np.minimum(x_ndvi[-1], x_map[-1])
            bottom = np.maximum(y_ndvi[-1], y_map[-1])
            r_ndvi = [
                np.round((left - x_ndvi[0]) / RES30).astype(np.int64),
                np.round((top - y_ndvi[0]) / -RES30).astype(np.int64),
                np.round((right - x_ndvi[0]) / RES30).astype(np.int64),
                np.round((bottom - y_ndvi[0]) / -RES30).astype(np.int64),
            ]
            r_map = [
                np.round((left - x_map[0]) / RES30).astype(np.int64),
                np.round((top - y_map[0]) / -RES30).astype(np.int64),
                np.round((right - x_map[0]) / RES30).astype(np.int64),
                np.round((bottom - y_map[0]) / -RES30).astype(np.int64),
            ]
            ndvi = ndvi[:, r_ndvi[0]:r_ndvi[2], r_ndvi[1]:r_ndvi[3]]
            ricemap = ricemap[0, r_map[0]:r_map[2], r_map[1]:r_map[3]]
            index = np.nonzero(ricemap)
            if index[0].size == 0:
                continue
            ndviseries = ndvi[:, index[0], index[1]]
            sio.savemat(outfile, {"arr_0": ndviseries}, do_compression=True)
            del ndvi, ricemap
            gc.collect()


if __name__ == "__main__":
    main()
