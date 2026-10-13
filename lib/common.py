#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared constants and helpers for the CCDPheno-Rice production pipeline.

All scripts in this repository accept the same set of data-directory
arguments (whose defaults are derived from ``--data-root``), so that a user
only needs to point ``--data-root`` (or the individual paths) at their own
data to run the whole pipeline.
"""
import argparse
from pathlib import Path

# Root directory of this repository.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Default data directory. All input/output data is expected under this
# directory unless overridden by the corresponding argument.
DEFAULT_DATA_ROOT = REPO_ROOT / "data"

# The eight rice phenological stages used throughout the pipeline.
PHENOS = [
    "transplanting", "regreening", "tillering", "stem_elongation",
    "booting", "heading", "milk_ripening", "maturity",
]

# Province names of the rice growing regions used by the pipeline.
PLACES = [
    "Jiangsu", "Sichuan", "Heilongjiang", "Yunnan", "Chongqing",
    "Guizhou", "Jilin", "Liaoning", "Henan", "Shanghai", "Shaanxi",
    "Shandong", "Hebei", "Inner_Mongolia", "Ningxia", "Tianjin",
    "Hunan", "Jiangxi", "Hubei", "Guangxi", "Anhui", "Zhejiang",
    "Fujian", "Guangdong", "Hainan",
]

# Provinces with only single-season rice (no early/late rice).
PLACES_SINGLE = [p for p in PLACES if p not in ("Guangdong", "Hainan")]

# Provinces with double-season (early + late) rice.
PLACES_DOUBLE = [
    "Hunan", "Jiangxi", "Hubei", "Guangxi", "Anhui",
    "Zhejiang", "Fujian", "Guangdong", "Hainan",
]

# Codes of the fused rice distribution map:
#   1 | 4 -> single-season rice,  2 -> double-season (early & late) rice.
SINGLE_RICE_CODES = (1, 4)
DOUBLE_RICE_CODE = 2

# Nominal resolution of the 30 m products (0.000269494585236 degrees).
RES30 = 0.000269494585236
# Resolution of the 10 m products.
RES10 = 0.000089831528412

# Days of the 46 eight-day NDVI composites within a year.
JDAYS = list(range(1, 361 + 8, 8))

# Feature column names of the 46 NDVI composites.
BAND_COLS = [f"{i}_NDVI" for i in range(46)]

def crop_types_for_place(place):
    """Return the rice types that grow in a province.

    Guangdong and Hainan have only double-season rice; the seven provinces of
    the middle-lower Yangtze region grow both single- and double-season rice;
    all other provinces have only single-season rice.
    """
    if place in ("Guangdong", "Hainan"):
        return ["early", "late"]
    if place in ("Zhejiang", "Anhui", "Hubei", "Hunan", "Jiangxi", "Fujian", "Guangxi"):
        return ["single", "early", "late"]
    return ["single"]


def rice_map_fname(place, yr, version=1):
    """Return the file name of the fused rice distribution map of a province.

    For Tianjin/Hebei (covered by CCD-Rice before 2017) the CCD-Rice base map
    is used; otherwise the China staple-crops map is used.
    """
    if place in ("Tianjin", "Hebei") and yr <= 2016:
        return f"CCD-Rice-{place}-{yr}-fusion_range-v{version}.tif"
    return f"China-staple-crops-{place}-{yr}-fusion_range-v{version}.tif"


def ndvi_fname(province, yr, jday):
    """Return the file name of a single 8-day NDVI composite.

    The file name follows the InENVI naming convention; adjust this function
    if your NDVI files are named differently.
    """
    return (
        f"{province}_P20WR10_{yr}{jday:03d}"
        f"_8Day_WGS30m_WWHD_Scale10000_V2_Uint8.tif"
    )


def model_fname(crop, pheno, buffer, w, fnumber, n_estimators, pixlim, fstation, do_offset):
    """Return the file name of a trained phenological-stage RF model."""
    name = f"{crop}-{pheno}-buffer-{buffer}"
    if do_offset:
        name += f"-offset2-w{w}"
    name += (
        f"-filter{fnumber}-n_estimators{n_estimators}"
        f"-pix{pixlim}-fstation{fstation}_3-all_data"
    )
    return name + ".joblib.z"


def add_data_root_arg(parser):
    """Add a common ``--data-root`` argument to an argparse parser."""
    parser.add_argument(
        "--data-root", type=Path, default=DEFAULT_DATA_ROOT,
        help=(
            "root directory that stores all input/output data of the pipeline "
            "(default: %(default)s). Individual directories can be overridden "
            "with the other --*-dir arguments."
        ),
    )
    return parser


def make_paths(args, obs_dirs=None, data_dirs=None):
    """Build the dictionary of data paths from an argparse namespace.

    ``obs_dirs`` and ``data_dirs`` are lists of directory names under
    ``args.data_root`` whose Path objects should be created and returned.
    """
    paths = {}
    for name in list(obs_dirs or []) + list(data_dirs or []):
        paths[name] = args.data_root / name
    return paths
