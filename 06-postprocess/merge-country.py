#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge the provincial phenological rasters into one national GeoTiff file per
(rice type, phenological stage, year).

The national products are named
``CCDPheno-Rice-China-<rice type>-<stage>-<year>-v<version>.tif``. Note that
the single-season rice extends over the whole country while the early/late
rice only covers the double-season region, so the two groups use different
spatial extents.

Example:
    python 06-postprocess/merge-country.py --version v5.4
"""
import argparse
import glob
import multiprocessing as mp
import os
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "lib"))
from common import (
    PHENOS, PLACES_DOUBLE, PLACES_SINGLE, RES30, add_data_root_arg,
)

CROPS = ["single", "early", "late"]

# Spatial extents (xmin ymax xmax ymin, in degrees) of the national rasters.
RNG_SINGLE = "97.172201587663238 53.560431848115066 135.0873947845160785 20.9030780092165855"
RNG_DOUBLE = "104.448016399860876 34.653769726308326 122.1395274368485673 17.9698989435180358"

# Tags of the corrected (and gap-filled) provincial rasters that are merged
# into the national products.
SRC_TAG_SINGLE = "v5-unified2-filled"
SRC_TAG_DOUBLE = "v5-unified2-el-filled"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge the provincial rasters into national products."
    )
    add_data_root_arg(parser)
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--version", default="v5.4",
                        help="version tag appended to the output file names.")
    parser.add_argument("--years-start", type=int, default=2001)
    parser.add_argument("--years-end", type=int, default=2024)
    parser.add_argument("--crops", nargs="+", default=CROPS, choices=CROPS)
    parser.add_argument("--gdal-merge-path", default=None,
                        help="path of the gdal_merge.py utility "
                             "(default: found in PATH).")
    parser.add_argument("--nproc", type=int, default=4)
    return parser.parse_args()


def merge(task):
    """Merge the provincial rasters of one (year, rice type, stage) pair."""
    yr, ricetype, pheno, version, result_dir, gdal_merge = task
    print(yr, ricetype, pheno)
    outfile = os.path.join(
        result_dir, f"CCDPheno-Rice-China-{ricetype}-{pheno}-{yr}-{version}.tif"
    )
    if os.path.isfile(outfile):
        return 0
    if ricetype == "single":
        flist = []
        for place in PLACES_SINGLE:
            # Tianjin/Hebei are covered by the single-season maps only
            # before 2017.
            if yr >= 2017 and place in ["Hebei", "Tianjin"]:
                continue
            file = os.path.join(
                result_dir,
                f"{place}-{yr}-{ricetype}-{pheno}-{SRC_TAG_SINGLE}.tif",
            )
            if os.path.isfile(file):
                flist.append(file)
    else:
        flist = glob.glob(
            f"{result_dir}/*-{yr}-{ricetype}-{pheno}-{SRC_TAG_DOUBLE}.tif"
        )
    if not flist:
        print(f"  no provincial files for {yr} {ricetype} {pheno}")
        return 0

    tempfile = os.path.join(
        result_dir, f"temp-{yr}-{ricetype}-{pheno}.txt"
    )
    with open(tempfile, "w") as f:
        for file in flist:
            f.write(f'"{file}"\n')

    rng = RNG_SINGLE if ricetype == "single" else RNG_DOUBLE
    res = RES30
    os.system(
        f"{gdal_merge} -ot Int16 -of GTiff -n -999 -a_nodata -999 "
        f"-ps {res} {res} -ul_lr {rng} "
        "-co COMPRESS=DEFLATE -co PREDICTOR=2 -co ZLEVEL=9 "
        "-co BIGTIFF=YES -co NUM_THREADS=80 "
        f"-o {outfile} --optfile {tempfile}"
    )
    os.remove(tempfile)
    return 0


def main():
    args = parse_args()
    result_dir = args.result_dir or (args.data_root / "result")
    gdal_merge = args.gdal_merge_path or shutil.which("gdal_merge.py")
    if gdal_merge is None:
        raise SystemExit("gdal_merge.py not found; install GDAL or pass "
                         "--gdal-merge-path.")

    years = range(args.years_start, args.years_end + 1)
    tasks = [
        (yr, ricetype, pheno, args.version, str(result_dir), gdal_merge)
        for yr in years for ricetype in args.crops for pheno in PHENOS
    ]
    pools = mp.Pool(args.nproc)
    pools.map(merge, tasks)
    pools.close()
    pools.join()


if __name__ == "__main__":
    main()
