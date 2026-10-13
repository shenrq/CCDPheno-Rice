#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Correct the double-season (early + late) rice phenological rasters.

Pixels whose interval between early-season maturity and late-season
transplanting is smaller than ``--min-gap`` days are repaired with the
early-late gap-RF model:

  * The target interval is predicted from the NDVI composites, then the late
    season is shifted as a whole so that
    ``late_transplanting - early_maturity = predicted_gap``. Early-season
    stages are kept unchanged, so the internal gaps of both seasons are
    preserved.
  * Pixels whose NDVI is invalid are set to 999 for all late-season stages.

Example:
    python 05-correct/correct-early-late-raster.py --in-version v5-unified2 --out-version v5-unified2-el
"""
import argparse
import gc
import multiprocessing as mp
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import rasterio
from sklearnex import patch_sklearn

patch_sklearn()
sys.path.append(str(Path(__file__).resolve().parents[1] / "lib"))
from common import (
    JDAYS, PHENOS, PLACES_DOUBLE, add_data_root_arg, ndvi_fname,
    rice_map_fname,
)
from raster_io import readgtiff, writegtiff

# Early-season maturity is the 8th stage (index 7); late-season
# transplanting is the 1st stage (index 0).
I_EARLY_MAT = 7
I_LATE_TRA = 0
NODATA = -999

# Module-level state inherited by the multiprocessing workers (fork).
EL_MODEL = None
CFG = {}


def process_place_year(args_place_yr):
    place, yr = args_place_yr
    result_dir = CFG["result_dir"]
    out_dir = result_dir

    outfiles = [
        os.path.join(out_dir, f"{place}-{yr}-{ct}-{ph}-{CFG['out_version']}.tif")
        for ct in ["early", "late"] for ph in PHENOS
    ]
    if all(os.path.isfile(f) for f in outfiles):
        return

    # Fused rice distribution map (double-season pixels).
    rice_fname = CFG["rice_dir"] / str(yr) / rice_map_fname(place, yr)
    if not os.path.isfile(rice_fname):
        return
    ricemap, ref = readgtiff(rice_fname)
    rice_double = ricemap[0] == 2
    valid_num = np.sum(rice_double)
    del ricemap
    gc.collect()

    early_files = [
        result_dir / f"{place}-{yr}-early-{ph}-{CFG['in_version']}.tif" for ph in PHENOS
    ]
    late_files = [
        result_dir / f"{place}-{yr}-late-{ph}-{CFG['in_version']}.tif" for ph in PHENOS
    ]
    if not all(os.path.isfile(f) for f in early_files + late_files):
        print(f"    skip {place} {yr}: some phenology files missing")
        return

    early_stack = np.concatenate([readgtiff(f)[0] for f in early_files])  # (8, H, W)
    late_stack = np.concatenate([readgtiff(f)[0] for f in late_files])    # (8, H, W)

    # Pixels whose rotation interval is too small.
    E0 = early_stack[I_EARLY_MAT]
    L0 = late_stack[I_LATE_TRA]
    valid = (E0 != NODATA) & (L0 != NODATA) & (E0 != 999) & (L0 != 999)
    wrong = (L0 - E0 < CFG["min_gap"]) & valid
    wrong_num = np.sum(wrong)
    print(f"  {place} {yr}: double-season {valid_num:,}, wrong {wrong_num:,}")

    if wrong_num == 0:
        # No correction needed; write the original values.
        for ct, stack in [("early", early_stack), ("late", late_stack)]:
            for pi, ph in enumerate(PHENOS):
                outfile = os.path.join(out_dir, f"{place}-{yr}-{ct}-{ph}-{CFG['out_version']}.tif")
                writegtiff(outfile, stack[pi], ref, nodata=NODATA, nthread=10)
        del early_stack, late_stack
        gc.collect()
        return

    # NDVI of the wrong pixels.
    rows, cols = np.nonzero(wrong)
    n_err = len(rows)
    ndvi_err = np.zeros((n_err, 46), dtype=np.uint8)
    ndvi_files = [
        CFG["ndvi_dir"] / place / ndvi_fname(place, yr, jday)
        for jday in JDAYS
    ]
    for j, ndvi_file in enumerate(ndvi_files):
        if not os.path.isfile(ndvi_file):
            continue
        with rasterio.open(ndvi_file) as ds:
            ndvi_err[:, j] = ds.read(1)[rows, cols]
        gc.collect()

    ndvi_valid = np.any(ndvi_err != 0, axis=1)
    n_valid_sel = np.sum(ndvi_valid)
    print(f"    pixels with predictable gap: {n_valid_sel}")

    # Only the NDVI-valid pixels are repaired.
    if n_valid_sel > 0:
        G = EL_MODEL.predict(ndvi_err[ndvi_valid]).astype(np.float32)
        G = np.maximum(CFG["min_gap"], G)

        # Shift the late season as a whole; early season is unchanged.
        E0_sel = early_stack[I_EARLY_MAT][rows[ndvi_valid], cols[ndvi_valid]].astype(np.float32)
        L0_sel = late_stack[I_LATE_TRA][rows[ndvi_valid], cols[ndvi_valid]].astype(np.float32)
        delta_l = np.int16(np.round(G - L0_sel + E0_sel))

        for pi in range(8):
            late_stack[pi, rows[ndvi_valid], cols[ndvi_valid]] += delta_l

    # NDVI-invalid pixels: all late-season stages are set to 999.
    if wrong_num > n_valid_sel:
        rows_bad = rows[~ndvi_valid]
        cols_bad = cols[~ndvi_valid]
        for pi in range(8):
            late_stack[pi, rows_bad, cols_bad] = 999
        print(f"    late season set to 999: {wrong_num - n_valid_sel}")
    del ndvi_err, rows, cols
    gc.collect()

    for ct, stack in [("early", early_stack), ("late", late_stack)]:
        stack = np.round(stack).astype(np.int16)
        for pi, ph in enumerate(PHENOS):
            outfile = os.path.join(out_dir, f"{place}-{yr}-{ct}-{ph}-{CFG['out_version']}.tif")
            writegtiff(outfile, stack[pi], ref, nodata=NODATA, nthread=10)
            print(f"    written: {outfile}")
    del early_stack, late_stack
    gc.collect()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Correct the double-season rice phenological rasters."
    )
    add_data_root_arg(parser)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--rice-dir", type=Path, default=None)
    parser.add_argument("--ndvi-dir", type=Path, default=None)
    parser.add_argument("--in-version", default="v5-unified2")
    parser.add_argument("--out-version", default="v5-unified2-el")
    parser.add_argument("--min-gap", type=int, default=1,
                        help="minimum allowed interval (days) between late "
                             "transplanting and early maturity.")
    parser.add_argument("--years-start", type=int, default=2001)
    parser.add_argument("--years-end", type=int, default=2024)
    parser.add_argument("--nproc", type=int, default=12)
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = args.model_dir or (args.data_root / "models")
    result_dir = args.result_dir or (args.data_root / "result")
    rice_dir = args.rice_dir or (args.data_root / "rice-map")
    ndvi_dir = args.ndvi_dir or (args.data_root / "ndvi")

    global EL_MODEL
    model_fp = os.path.join(model_dir, "early-late-gap-NDVI46-RF.joblib.z")
    print(f"loading model: {model_fp}")
    EL_MODEL = joblib.load(model_fp)

    CFG.update({
        "in_version": args.in_version,
        "out_version": args.out_version,
        "min_gap": args.min_gap,
        "result_dir": result_dir,
        "rice_dir": rice_dir,
        "ndvi_dir": ndvi_dir,
    })

    years = list(range(args.years_start, args.years_end + 1))
    for place in PLACES_DOUBLE:
        task_list = [(place, yr) for yr in years]
        with mp.Pool(args.nproc) as pool:
            pool.map(process_place_year, task_list)

    print("\ndone!")


if __name__ == "__main__":
    main()
