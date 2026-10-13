#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas
"""
Estimate rice phenological dates of the pixels around AMS sites with the
SMF-S method, and save the offsets (SMF-S date minus observed date).

The SMF-S implementation is provided by the ``SMFS-RF`` repository which is
included as a git submodule under ``external/SMFS-RF`` (its nested submodule
``SMF_S_Release`` provides the ``smf_s_class`` module).

This is Step 2 of the pipeline. It must be run for each rice type
(``--crop single|early|late``), after ``extract-sites-rice-NDVI.py``.

Example:
    python 01-site-curve/site-curve-offset-SMF-S.py --crop single
"""
import argparse
import multiprocessing as mp
import os
import re
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import scipy.io as sio

sys.path.append(str(Path(__file__).resolve().parents[1] / "lib"))
from common import (
    PHENOS, add_data_root_arg,
)

# The SMF-S source is shipped inside the SMFS-RF submodule. Override with
# --smfs-src if the submodule is located elsewhere.
DEFAULT_SMFS_SRC = (
    Path(__file__).resolve().parents[1] / "external" / "SMFS-RF" / "SMF_S_Release" / "src"
)

# Module-level configuration. The multiprocessing workers (fork) inherit it.
CFG = {}


def read_curves(file):
    curves = sio.loadmat(os.path.join(CFG["path"], file))["arr_0"]
    return curves, curves.shape[1]


def cut_X(X):
    crop, w = CFG["crop"], CFG["w"]
    # Keep only the NDVI values of the growing season.
    if crop == "single":
        X[:max(0, int(9 - w / 8))] = X[max(0, int(9 - w / 8))]
    elif crop == "early":
        X[min(46, int(29 + w / 8)):] = X[min(46, int(29 + w / 8))]
    elif crop == "late":
        X[:max(0, int(22 - w / 8))] = X[max(0, int(22 - w / 8))]
    return X


def get_offset(curves, number, y):
    """Return the offsets (SMF-S date minus observed date) of one station."""
    from smf_s_class import SMFS
    if number == 1:
        return np.array([0])
    smfs_model = SMFS(np.mean(curves, axis=1), y, CFG["doys"])
    smfs_model = cut_X(smfs_model)
    smfs_model.WIN = CFG["w"]
    offset = np.array(
        [smfs_model.doit(cut_X(curves[:, j])) for j in range(curves.shape[1])]
    )
    offset[offset == 0] = np.nan
    offset -= y
    return offset


def write_offset(file_y):
    """Compute the SMF-S offsets of one station and write them to a .npz file."""
    file, y = file_y
    pheno, w = CFG["pheno"], CFG["w"]
    outfile = os.path.join(
        CFG["path"], re.sub("NDVI-time_series", f"y_offset-SMF-S-{pheno}-w{w}", file)
    )[:-4]
    if os.path.isfile(outfile + ".npz"):
        return 1
    offset = get_offset(*read_curves(file), y)
    np.savez_compressed(outfile, offset)
    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate SMF-S offsets of rice phenological dates around AMS sites."
    )
    add_data_root_arg(parser)
    parser.add_argument(
        "--crop", required=True, choices=["single", "early", "late"],
        help="rice type: single-season, double-season early, or double-season late.",
    )
    parser.add_argument("--obs-dir", type=Path, default=None)
    parser.add_argument("--obs-file", type=Path, default=None)
    parser.add_argument("--years-start", type=int, default=2000)
    parser.add_argument("--years-end", type=int, default=2015)
    parser.add_argument("--buffer", type=float, default=0.05,
                        help="buffer size in degrees (must match the one used "
                             "by extract-sites-rice-NDVI.py).")
    parser.add_argument("--w", type=int, default=8,
                        help="width (in days) of the SMF-S window.")
    parser.add_argument("--rice-map-version", type=int, default=1)
    parser.add_argument("--sites-dir", type=Path, default=None)
    parser.add_argument("--smfs-src", type=Path, default=DEFAULT_SMFS_SRC)
    parser.add_argument("--nproc", type=int, default=60)
    return parser.parse_args()


def main():
    args = parse_args()
    sys.path.append(str(args.smfs_src))

    obs_dir = args.obs_dir or (args.data_root / "observation")
    sites_dir = args.sites_dir or (args.data_root / "sites-curve")
    obs_file = args.obs_file or (
        obs_dir / f"{args.crop}-rice-phenology.xlsx"
    )

    daystep = 8
    doys = np.arange(1, 361 + daystep, daystep)
    CFG.update({
        "crop": args.crop,
        "w": args.w,
        "doys": doys,
        "path": str(sites_dir / args.crop),
        "pheno": None,
    })
    buffer = args.buffer
    vmap = args.rice_map_version
    sheet0 = pd.read_excel(obs_file)
    for pheno in PHENOS:
        print(args.crop, pheno)
        CFG["pheno"] = pheno
        sheet = sheet0.copy()
        sheet.reset_index(drop=True, inplace=True)
        flist = os.listdir(CFG["path"])

        flist2 = []
        deletion = []
        for i, row in sheet.iterrows():
            station = row["station"]
            yr = row["year"]
            file = f"{station}-{yr}-NDVI-time_series-v{vmap}-buffer-{buffer}.mat"
            if file in flist:
                flist2.append(file)
            else:
                deletion.append(i)
        sheet.drop(deletion, inplace=True)
        sheet.reset_index(drop=True, inplace=True)

        y0 = sheet[pheno].to_numpy()
        flist3 = [file for i, file in enumerate(flist2) if not np.isnan(y0[i])]
        y0 = list(y0[~np.isnan(y0)])

        pools = mp.Pool(processes=args.nproc)
        status = pools.map(write_offset, zip(flist3, y0))
        pools.close()
        pools.join()


if __name__ == "__main__":
    main()
