#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predict the phenological dates of every rice pixel of China with the trained
RF models, and write the 30 m phenological-stage rasters.

For each (province, year, rice type), the rice pixels are located from the
fused rice distribution map, their 46 8-day NDVI composites are loaded, and
the RF model of each of the 8 phenological stages predicts the DOY. Pixels
whose NDVI is invalid (all composites are zero) are set to 999.

This is Step 4 of the pipeline (outputs the raw ``v5`` products).

Example:
    python 03-predict/get-pheno-raster.py
"""
import argparse
import gc
import multiprocessing as mp
import os
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearnex import patch_sklearn

patch_sklearn()
sys.path.append(str(Path(__file__).resolve().parents[1] / "lib"))
from common import (
    BAND_COLS, JDAYS, PHENOS, PLACES, SINGLE_RICE_CODES,
    DOUBLE_RICE_CODE, add_data_root_arg, crop_types_for_place, model_fname,
    ndvi_fname, rice_map_fname,
)
from raster_io import gtiffref, readgtiff, writegtiff

CROPS = ["single", "early", "late"]

# Module-level state inherited by the multiprocessing workers (fork).
MODELS = {}


def predict_batch(args_p):
    """Predict the phenological dates of one batch of pixels."""
    crop, pheno, batch = args_p
    invalid = np.all(batch == 0, axis=1)
    predicted0 = np.round(
        MODELS[crop][pheno].predict(batch)
    ).astype(np.int16)
    predicted0[invalid] = 999
    return predicted0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict rice phenological dates for every rice pixel of China."
    )
    add_data_root_arg(parser)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--rice-dir", type=Path, default=None,
                        help="directory of the fused rice distribution maps "
                             "(one subdirectory per year).")
    parser.add_argument("--ndvi-dir", type=Path, default=None,
                        help="directory of the 8-day NDVI product (one "
                             "subdirectory per province).")
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--years-start", type=int, default=2001)
    parser.add_argument("--years-end", type=int, default=2024)
    parser.add_argument("--rice-map-version", type=int, default=1)
    parser.add_argument("--out-version", default="v5",
                        help="version tag appended to the output file names.")
    # The following model parameters must match the ones used by
    # train-with-all-data-filter-station.py.
    parser.add_argument("--model-buffer", type=float, default=0.05)
    parser.add_argument("--model-w", type=int, default=8)
    parser.add_argument("--model-filter", type=int, default=0)
    parser.add_argument("--model-n-estimators", type=int, default=100)
    parser.add_argument("--model-pixlim", type=int, default=100)
    parser.add_argument("--model-fstation", type=float, default=2)
    parser.add_argument("--model-offset2", action="store_true", default=True)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--nproc", type=int, default=80)
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = args.model_dir or (args.data_root / "models")
    rice_dir = args.rice_dir or (args.data_root / "rice-map")
    ndvi_dir = args.ndvi_dir or (args.data_root / "ndvi")
    result_dir = args.result_dir or (args.data_root / "result")
    result_dir.mkdir(parents=True, exist_ok=True)

    # Load the RF model of every (rice type, phenological stage).
    global MODELS
    MODELS = {}
    for crop in CROPS:
        MODELS[crop] = {}
        for pheno in PHENOS:
            fp = os.path.join(
                model_dir,
                model_fname(crop, pheno, args.model_buffer, args.model_w,
                            args.model_filter, args.model_n_estimators,
                            args.model_pixlim, args.model_fstation,
                            args.model_offset2),
            )
            if os.path.isfile(fp):
                MODELS[crop][pheno] = joblib.load(fp)
            else:
                MODELS[crop][pheno] = None
                print(f"[warning] model not found: {fp}")
    print("models loaded.")

    def get_ndvi(file, valid):
        ndvi, _ = readgtiff(file)
        return ndvi[0, :, :][valid]

    years = range(args.years_start, args.years_end + 1)
    for place in PLACES:
        for yr in years:
            rice_fname = rice_dir / str(yr) / rice_map_fname(place, yr, args.rice_map_version)
            ndvifiles = [
                ndvi_dir / place / ndvi_fname(place, yr, jday)
                for jday in JDAYS
            ]
            outfiles = [
                result_dir / f"{place}-{yr}-{crop}-{pheno}-{args.out_version}.tif"
                for crop in CROPS for pheno in PHENOS
            ]
            if all([os.path.isfile(of) for of in outfiles]):
                ref1 = gtiffref(outfiles[0])
                ref2 = gtiffref(ndvifiles[0])
                if ref1["width"] == ref2["width"] and ref1["height"] == ref2["height"]:
                    continue

            ricemap, ref = readgtiff(rice_fname)
            valid = {}
            valid["single"] = (
                (ricemap[0, :, :] == SINGLE_RICE_CODES[0]) |
                (ricemap[0, :, :] == SINGLE_RICE_CODES[1])
            )
            valid["early"] = valid["late"] = ricemap[0, :, :] == DOUBLE_RICE_CODE
            del ricemap

            for crop in CROPS:
                if crop not in crop_types_for_place(place):
                    continue
                if MODELS[crop][PHENOS[0]] is None:
                    print(f"  skip {place} {yr} {crop}: model missing")
                    continue
                validnum = np.sum(valid[crop])
                if validnum == 0:
                    continue
                gc.collect()
                index = np.nonzero(valid[crop])
                ndvi = np.vstack([
                    get_ndvi(f, valid[crop]) for f in ndvifiles
                ]).T
                print(place, yr, crop)

                bsize = args.batch_size
                batches = [
                    ndvi[i * bsize:min(validnum, (i + 1) * bsize), :]
                    for i in range(validnum // bsize + 1)
                ]
                del ndvi

                for pheno in PHENOS:
                    print(pheno)
                    outfile = result_dir / f"{place}-{yr}-{crop}-{pheno}-{args.out_version}.tif"

                    pools = mp.Pool(args.nproc)
                    predicted = pools.map(
                        predict_batch, [(crop, pheno, b) for b in batches]
                    )
                    pools.close()
                    pools.join()

                    predicted = np.hstack(predicted)
                    pheno_raster = np.ones_like(valid[crop], dtype=np.int16) * -999
                    pheno_raster[index[0], index[1]] = predicted
                    writegtiff(outfile, pheno_raster, ref, nodata=-999, nthread=50)
                    del predicted, pheno_raster
                del batches, index
            del valid


if __name__ == "__main__":
    main()
