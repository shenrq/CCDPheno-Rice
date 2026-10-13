#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fill the pixels whose phenological date could not be estimated.

Phenological rasters may contain pixels set to 999 (NDVI invalid at
prediction/correction time). They are filled with the mean of the valid
pixels of a surrounding block. The block size starts at
``--block-start`` pixels (30 m) and increases until every pixel is filled.

The script processes every file of the form ``*-<tag>.tif`` in the result
directory. Run it once per version tag that needs filling:

    python 06-postprocess/fill-failed-pheno.py --tag v5-unified2       # single
    python 06-postprocess/fill-failed-pheno.py --tag v5-unified2-el    # early/late
"""
import argparse
import glob
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "lib"))
from common import add_data_root_arg
from raster_io import readgtiff, writegtiff


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fill failed (999) pixels with surrounding block means."
    )
    add_data_root_arg(parser)
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--tag", required=True,
                        help="version tag of the files to process, e.g. "
                             "v5-unified2 or v5-unified2-el.")
    parser.add_argument("--block-start", type=int, default=30,
                        help="initial block size in pixels.")
    parser.add_argument("--block-step", type=int, default=30,
                        help="block size increment per iteration.")
    parser.add_argument("--nproc", type=int, default=4)
    return parser.parse_args()


def fill_failed(data, block):
    width, height = data.shape[2], data.shape[1]
    for ix in range(width // block + 1):
        for iy in range(height // block + 1):
            temp = data[
                0, iy * block:min(iy * block + block, height),
                ix * block:min(ix * block + block, width)
            ]
            failed = temp == 999
            if np.sum(failed) == 0:
                continue
            valid = ~failed & (temp != -999)
            if np.sum(valid) == 0:
                continue
            temp[failed] = np.round(np.mean(temp[valid])).astype(np.uint16)
            data[
                0, iy * block:min(iy * block + block, height),
                ix * block:min(ix * block + block, width)
            ] = temp
    return data


def fill_p(args_file):
    file, args = args_file
    outfile = file[:-4] + "-filled.tif"
    if os.path.isfile(outfile):
        return 0
    print(file)

    data, ref = readgtiff(file)
    block = args.block_start
    while np.sum(data == 999) != 0:
        data = fill_failed(data, block)
        block += args.block_step
    print(f"  block size used: {block}")
    writegtiff(outfile, data, ref, nthread=20, nodata=-999)
    return 0


def main():
    args = parse_args()
    result_dir = args.result_dir or (args.data_root / "result")

    # Files that end with the exact tag (so that e.g. v5-unified2 does not
    # match the v5-unified2-el files) and have not been filled yet.
    flist = [
        f for f in glob.glob(f"{result_dir}/*.tif")
        if f.endswith(f"-{args.tag}.tif") and "filled" not in f
    ]
    print(f"{len(flist)} files to process")

    pools = mp.Pool(args.nproc)
    pools.map(fill_p, [(f, args) for f in flist])
    pools.close()
    pools.join()


if __name__ == "__main__":
    main()
