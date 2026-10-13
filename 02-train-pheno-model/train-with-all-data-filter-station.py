#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train the final phenological-stage Random Forest (RF) models with all
station data.

For each rice type x phenological stage, an RF regression model is trained
with the full NDVI time series (46 features) of the rice pixels around all
AMS sites as input and the observed phenological date as target. Optionally
(--do-offset2), the SMF-S offsets computed by site-curve-offset-SMF-S.py are
added to the labels, which aligns the pixel-level labels to the
knowledge-guided SMF-S estimates.

This is Step 3 of the pipeline. It trains 24 models (3 rice types x 8 stages).

Example:
    python 02-train-pheno-model/train-with-all-data-filter-station.py
"""
import argparse
import multiprocessing as mp
import os
import re
import sys
from pathlib import Path

import pandas as pd
import joblib
import numpy as np
import scipy.io as sio
from scipy.stats import norm
from sklearnex import patch_sklearn
from sklearn.ensemble import RandomForestRegressor

patch_sklearn()
sys.path.append(str(Path(__file__).resolve().parents[1] / "lib"))
from common import (  # noqa: E402
    PHENOS, add_data_root_arg, model_fname,
)

CROPS = ["single", "early", "late"]

# Module-level state inherited by the multiprocessing workers (fork).
SHEETS = {}   # crop -> observation table
CFG = {}      # training configuration


def read_curves(file, crop):
    curves = sio.loadmat(os.path.join(CFG["path"], crop, file))["arr_0"]
    return curves, curves.shape[1]


def read_offsets2(file, crop, number):
    offsets = {}
    for pheno in CFG["phenos"]:
        file2 = os.path.join(
            CFG["path"], crop,
            re.sub("NDVI-time_series", f"y_offset-SMF-S-{pheno}-w{CFG['w']}", file),
        )[:-4] + ".npz"
        if not os.path.isfile(file2):
            offset = np.ones(number) * np.nan
        else:
            offset = np.load(os.path.join(CFG["path"], crop, file2))["arr_0"]
        offset[(offset > 16) | (offset < -16)] = np.nan
        offsets[pheno] = offset
    return offsets


def read_curves_offsets2(file, crop):
    curves, number = read_curves(file, crop)
    offsets = read_offsets2(file, crop, number)
    return curves, offsets, number


def filter_curves(curves, offsets, number, fnumber, pixlim):
    if curves.shape[1] < pixlim:
        return curves, number, offsets
    mean_curve = np.mean(curves, axis=1).reshape(-1, 1)
    std_curve = np.std(curves, axis=1).reshape(-1, 1)
    select = np.sum(
        (curves < mean_curve + std_curve) &
        (curves > mean_curve - std_curve), axis=0
    ) >= fnumber * curves.shape[0]
    rest = np.nonzero(~select)[0]
    select = np.nonzero(select)[0]
    number = select.size
    if number < pixlim:
        select2 = np.random.choice(rest.size, size=pixlim - number, replace=False)
        select = np.concatenate([select, select2])
    else:
        select = select[np.random.choice(number, size=pixlim, replace=False)]
    for pheno in CFG["phenos"]:
        offsets[pheno] = offsets[pheno][select]
    return curves[:, select], pixlim, offsets


def get_y(y0, nums):
    y = np.zeros([np.sum(nums)])
    n = 0
    for i, num in enumerate(nums):
        y[n:n + num] = y0[i]
        n += num
    return y


def get_sample_p(y0, outlier):
    mu = np.mean(y0)
    sigma = np.std(y0)
    y1 = np.minimum(y0, mu + outlier * sigma)
    y1 = np.maximum(y1, mu - outlier * sigma)
    p = 1 / norm.pdf(y1, mu, sigma)
    p = p / np.sum(p)
    return p


def train_model(crop, pheno):
    sheet = SHEETS[crop].copy()
    sheet.reset_index(drop=True, inplace=True)
    flist = os.listdir(os.path.join(CFG["path"], crop))

    flist2 = []
    deletion = []
    for i, row in sheet.iterrows():
        station = row["station"]
        yr = row["year"]
        file = f"{station}-{yr}-NDVI-time_series-v{CFG['vmap']}-buffer-{CFG['buffer']}.mat"
        if file in flist:
            flist2.append(file)
        else:
            deletion.append(i)
    sheet.drop(deletion, inplace=True)
    sheet.reset_index(drop=True, inplace=True)

    y0 = sheet[pheno].to_numpy()
    flist3 = [file for i, file in enumerate(flist2) if not np.isnan(y0[i])]
    y0 = y0[~np.isnan(y0)]

    # Sample 3x the number of station-years with a probability that
    # down-weights stations with outlying observations.
    p = get_sample_p(y0, CFG["fstation"])
    itrain = np.random.choice(
        range(y0.size), int(np.round(y0.size * 3)), p=p
    )
    f_train = [flist3[i] for i in itrain]
    y0_train = y0[itrain]

    modelfile = os.path.join(
        CFG["model_dir"],
        model_fname(crop, pheno, CFG["buffer"], CFG["w"], CFG["fnumber"],
                    CFG["n_estimators"], CFG["pixlim"], CFG["fstation"],
                    CFG["do_offset2"]),
    )
    curves_train = [
        filter_curves(*read_curves_offsets2(file, crop), CFG["fnumber"], CFG["pixlim"])
        for file in f_train
    ]
    X_train = np.concatenate([curve[0] for curve in curves_train], axis=1)
    nums = [curve[1] for curve in curves_train]
    if CFG["do_offset2"]:
        y_train = get_y(y0_train, nums) + np.concatenate(
            [curve[2][pheno] for curve in curves_train]
        )
        y_valid = ~np.isnan(y_train)
        y_train = y_train[y_valid]
        X_train = X_train[:, y_valid]
    else:
        y_train = get_y(y0_train, nums)

    model = RandomForestRegressor(n_jobs=10, n_estimators=CFG["n_estimators"])
    model.fit(X_train.T, y_train)
    joblib.dump(model, modelfile)
    print(f"model saved: {modelfile}")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the final phenological-stage RF models with all data."
    )
    add_data_root_arg(parser)
    parser.add_argument("--obs-dir", type=Path, default=None)
    parser.add_argument("--years-start", type=int, default=2000)
    parser.add_argument("--years-end", type=int, default=2015)
    parser.add_argument("--rice-map-version", type=int, default=1,
                        help="version number embedded in the curve file names.")
    parser.add_argument("--sites-dir", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--buffer", type=float, default=0.05)
    parser.add_argument("--w", type=int, default=8,
                        help="width (in days) of the SMF-S window.")
    parser.add_argument("--fnumber", type=int, default=0,
                        help="fraction (x 46) of 8-day composites that a pixel "
                             "must stay within the mean +/- std envelope to be "
                             "kept by the curve filter.")
    parser.add_argument("--pixlim", type=int, default=100,
                        help="maximum number of pixels sampled per station.")
    parser.add_argument("--fstation", type=float, default=2,
                        help="number of standard deviations used by the "
                             "station-level label sampling probability.")
    parser.add_argument("--do-offset2", action="store_true", default=True,
                        help="add SMF-S offsets to the training labels.")
    parser.add_argument("--no-offset2", dest="do_offset2", action="store_false",
                        help="train with observed dates only.")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--nproc", type=int, default=None,
                        help="number of parallel processes (default: one per model).")
    return parser.parse_args()


def main():
    args = parse_args()
    obs_dir = args.obs_dir or (args.data_root / "observation")
    sites_dir = args.sites_dir or (args.data_root / "sites-curve")
    model_dir = args.model_dir or (args.data_root / "models")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Observation tables of the three rice types.
    for crop in CROPS:
        obs_file = obs_dir / f"{crop}-rice-phenology.xlsx"
        SHEETS[crop] = pd.read_excel(obs_file)

    CFG.update({
        "path": str(sites_dir),
        "w": args.w,
        "phenos": PHENOS,
        "vmap": args.rice_map_version,
        "buffer": args.buffer,
        "fnumber": args.fnumber,
        "pixlim": args.pixlim,
        "fstation": args.fstation,
        "do_offset2": args.do_offset2,
        "n_estimators": args.n_estimators,
        "model_dir": str(model_dir),
    })

    crop_phenos = [(crop, pheno) for crop in CROPS for pheno in PHENOS]
    pools = mp.Pool(args.nproc or len(crop_phenos))
    status = pools.starmap(train_model, crop_phenos)
    pools.close()
    pools.join()
    print(status)


if __name__ == "__main__":
    main()
