#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Correct the single-season rice phenological rasters.

Each of the 8 phenological stages is predicted independently, so the sequence
of a pixel may not be strictly increasing. This script detects such
inconsistent pixels and repairs them with the gap-RF models:

  * Local fix: for each run of consecutive inconsistent intervals, the
    involved stages are rebuilt from the predicted gaps and aligned to their
    original local mean.
  * Global fix (used when a run covers 3+ intervals, when the gap prediction
    is missing, or when a local fix creates a new inconsistency): all 8
    stages are rebuilt from the 7 predicted gaps and aligned to the original
    overall mean.

All correction follows the mean-alignment strategy: the relative sequence is
reconstructed from the gaps, then shifted so that its mean equals the mean of
the original (uncorrected) values.

Example:
    python 05-correct/correct-pheno-raster.py --in-version v5 --out-version v5-unified2
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
    JDAYS, PHENOS, PLACES, add_data_root_arg, crop_types_for_place,
    ndvi_fname, rice_map_fname,
)
from raster_io import gtiffref, readgtiff, writegtiff

GAP_NAMES = [
    "transplanting_to_regreening",
    "regreening_to_tillering",
    "tillering_to_stem_elongation",
    "stem_elongation_to_booting",
    "booting_to_heading",
    "heading_to_milk_ripening",
    "milk_ripening_to_maturity",
]
NODATA1 = -999
NODATA2 = 999

# Module-level state inherited by the multiprocessing workers (fork).
GAP_MODELS = {}
CFG = {}


def is_monotonic_increasing(arr, axis=0, min_gap=2):
    return np.all(np.diff(arr, axis=axis) >= min_gap, axis=axis)


def process_place_year(args_place_yr):
    place, yr = args_place_yr
    croptypes = crop_types_for_place(place)
    result_dir = CFG["result_dir"]
    out_dir = result_dir

    if all(
        os.path.isfile(os.path.join(out_dir, f"{place}-{yr}-{ct}-{ph}-{CFG['out_version']}.tif"))
        for ph in PHENOS for ct in croptypes
    ):
        return

    # Fused rice distribution map of the province and year.
    rice_fname = CFG["rice_dir"] / str(yr) / rice_map_fname(place, yr)
    if not os.path.isfile(rice_fname):
        return
    ricemap, ref = readgtiff(rice_fname)
    rice_single = (ricemap[0] == 1) | (ricemap[0] == 4)
    rice_double = ricemap[0] == 2
    del ricemap
    gc.collect()

    ndvi_files = [
        CFG["ndvi_dir"] / place / ndvi_fname(place, yr, jday)
        for jday in JDAYS
    ]

    for ct in croptypes:
        valid = rice_single if ct == "single" else rice_double
        valid_num = np.sum(valid)
        if valid_num == 0:
            continue

        pheno_files = [
            result_dir / f"{place}-{yr}-{ct}-{ph}-{CFG['in_version']}.tif"
            for ph in PHENOS
        ]
        outfiles = [
            result_dir / f"{place}-{yr}-{ct}-{ph}-{CFG['out_version']}.tif"
            for ph in PHENOS
        ]
        if all(os.path.isfile(f) for f in outfiles):
            ref_out = gtiffref(outfiles[0])
            ref_ndvi = gtiffref(ndvi_files[0])
            if ref_out["width"] != ref_ndvi["width"] or ref_out["height"] != ref_ndvi["height"]:
                for pi in range(8):
                    os.remove(outfiles[pi])
            else:
                continue
        if not all(os.path.isfile(f) for f in pheno_files):
            print(f"    skip {place} {yr} {ct}: some phenology files missing")
            continue

        data_stack = np.concatenate([readgtiff(f)[0] for f in pheno_files])  # (8, H, W)

        # Valid pixels whose sequence is not strictly increasing.
        valid_mask = (data_stack[0] != NODATA1) & (data_stack[0] != NODATA2)
        wrong = ~is_monotonic_increasing(data_stack, axis=0, min_gap=CFG["min_inner_gap"]) & valid_mask
        wrong_num = np.sum(wrong)
        print(f"  {place} {yr} {ct}: valid {valid_num:,}, wrong {wrong_num:,}")

        if wrong_num == 0:
            continue

        # NDVI of the wrong pixels (46 composites).
        ndvi_err = np.zeros((wrong_num, 46), dtype=np.uint8)
        for j, ndvi_file in enumerate(ndvi_files):
            if not os.path.isfile(ndvi_file):
                continue
            with rasterio.open(ndvi_file) as ds:
                ndvi_err[:, j] = ds.read(1)[wrong]
            gc.collect()

        ndvi_valid = np.any(ndvi_err != 0, axis=1)
        # Gap-RF predictions for the 7 intervals.
        gap_preds = np.ones((wrong_num, 7), dtype=np.float32) * np.nan
        for gi, gn in enumerate(GAP_NAMES):
            model = GAP_MODELS[ct][gn]
            if model is None:
                continue
            if np.sum(ndvi_valid) == 0:
                continue
            preds = model.predict(ndvi_err[ndvi_valid])
            gap_preds[ndvi_valid, gi] = np.maximum(CFG["min_inner_gap"], preds)
        del ndvi_err
        gc.collect()

        # ---- local-first / switch-to-global repair ----
        Xw = data_stack[:, wrong].astype(np.float32).T   # (nw, 8)
        nw = Xw.shape[0]
        inv_mat = np.diff(Xw.T, axis=0) < CFG["min_inner_gap"]   # (7, nw)
        # Pixels with any nodata / invalid stage are not locally repaired.
        nodata_hit = np.any(
            (Xw == NODATA1) | (Xw == NODATA2) | ~np.isfinite(Xw), axis=1
        )

        corrected = Xw.copy()
        for p in range(nw):
            m = inv_mat[:, p]
            g7 = gap_preds[p]
            global_fix = False
            if ndvi_valid[p] and not nodata_hit[p]:
                # Split the inverted intervals into consecutive runs.
                runs = []
                inrun = False
                for i, v in enumerate(m):
                    if v and not inrun:
                        start = i
                        inrun = True
                    elif not v and inrun:
                        runs.append((start, i - 1))
                        inrun = False
                if inrun:
                    runs.append((start, 6))
                if any((b - a + 1) >= 3 for a, b in runs):
                    global_fix = True   # a run covers 3+ intervals
                elif any(np.isnan(g7[a:b + 1]).any() for a, b in runs):
                    global_fix = True   # the run involves missing gap predictions
            else:
                global_fix = True       # NDVI invalid or nodata present

            if global_fix:
                # Rebuild the whole sequence and align to the original mean.
                if ndvi_valid[p]:
                    rel = np.concatenate(([0.0], np.cumsum(g7[:7])))
                else:
                    rel = np.zeros(8)
                off = np.mean(Xw[p]) - np.mean(rel)
                corrected[p] = rel + off
                continue

            # Local fix per run, each aligned to its own original mean.
            orig8 = Xw[p].copy()
            for a, b in runs:
                L = b - a + 1
                di = list(range(a, b + 2))     # the L+1 involved stages
                orig_local = Xw[p, di].copy()
                relL = np.zeros(L + 1)
                for k in range(L):
                    relL[k + 1] = relL[k] + g7[a + k]
                Xw[p, di] = relL + (np.mean(orig_local) - np.mean(relL))
            # A new inversion after the local fix: roll back and go global.
            if np.any(np.diff(Xw[p]) < CFG["min_inner_gap"]):
                Xw[p] = orig8
                rel = np.concatenate(([0.0], np.cumsum(g7[:7])))
                off = np.mean(Xw[p]) - np.mean(rel)
                corrected[p] = rel + off
            else:
                corrected[p] = Xw[p]

        del Xw, gap_preds, inv_mat
        gc.collect()

        corrected[np.isnan(corrected)] = NODATA2
        corrected = corrected.round().astype(np.int16)

        for pi, ph in enumerate(PHENOS):
            outfile = os.path.join(out_dir, f"{place}-{yr}-{ct}-{ph}-{CFG['out_version']}.tif")
            data_stack[pi, wrong] = corrected[:, pi]
            writegtiff(outfile, data_stack[pi], ref, nodata=NODATA1, nthread=10)
            print(f"    written: {outfile}")
        del corrected, data_stack
        gc.collect()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Correct the single-season rice phenological rasters."
    )
    add_data_root_arg(parser)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--rice-dir", type=Path, default=None)
    parser.add_argument("--ndvi-dir", type=Path, default=None)
    parser.add_argument("--in-version", default="v5")
    parser.add_argument("--out-version", default="v5-unified2")
    parser.add_argument("--min-inner-gap", type=int, default=2,
                        help="intervals smaller than this (days) are treated "
                             "as inversions.")
    parser.add_argument("--years-start", type=int, default=2001)
    parser.add_argument("--years-end", type=int, default=2024)
    parser.add_argument("--nproc", type=int, default=12,
                        help="number of processes used for each province.")
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = args.model_dir or (args.data_root / "models")
    result_dir = args.result_dir or (args.data_root / "result")
    rice_dir = args.rice_dir or (args.data_root / "rice-map")
    ndvi_dir = args.ndvi_dir or (args.data_root / "ndvi")

    # Preload the gap-RF models (7 gaps x 3 rice types).
    print("loading gap-RF models...")
    for ct in ["single", "early", "late"]:
        GAP_MODELS[ct] = {}
        for gn in GAP_NAMES:
            fp = os.path.join(model_dir, f"{ct}-{gn}-NDVI46-RF.joblib.z")
            if os.path.isfile(fp):
                GAP_MODELS[ct][gn] = joblib.load(fp)
            else:
                GAP_MODELS[ct][gn] = None
                print(f"  [warning] model not found: {fp}")

    CFG.update({
        "in_version": args.in_version,
        "out_version": args.out_version,
        "min_inner_gap": args.min_inner_gap,
        "model_dir": str(model_dir),
        "result_dir": result_dir,
        "rice_dir": rice_dir,
        "ndvi_dir": ndvi_dir,
    })

    years = list(range(args.years_start, args.years_end + 1))
    for place in PLACES:
        task_list = [(place, yr) for yr in years]
        with mp.Pool(args.nproc) as pool:
            pool.map(process_place_year, task_list)

    print("\ndone!")


if __name__ == "__main__":
    main()
