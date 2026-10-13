# Codes to generate CCDPheno-Rice product

This repository contains the full production pipeline of **CCDPheno-Rice**, a 30 m rice phenological dates dataset of China (2001–2024) covering three rice types — single-season rice, double-season early rice, and double-season late rice — and eight phenological stages (transplanting, regreening, tillering, stem elongation, booting, heading, milk ripening, and maturity). This code can be adapted to generate crop phenology datasets for other crops with a few modifications.

The core idea of the pipeline is:

1. Random forest (RF) regression models predict each phenological stage
   independently from the 46 eight-day 30 m NDVI composites.
2. Because the stages are predicted independently, some pixels get
   non-increasing (inverted) phenological sequences. "gap-RF" models that
   predict the intervals between adjacent stages from the NDVI composites are
   trained to repair such pixels.
3. Double-season rice needs an additional repair of the rotation interval
   between early-season maturity and late-season transplanting, performed by
   a dedicated early-late gap-RF model.

All corrections follow a *mean-alignment* strategy: the relative sequence is reconstructed from the predicted intervals, then shifted so that its mean matches the mean of the original (uncorrected) values.

## Repository structure

```
CCDPheno-source/
├── 01-site-curve/                    Step 1: station-level data
│   ├── extract-sites-rice-NDVI.py      extract NDVI curves around AMS sites
│   └── site-curve-offset-SMF-S.py      SMF-S offsets of pixel-level labels
├── 02-train-pheno-model/
│   └── train-with-all-data-filter-station.py   train the 24 stage RF models
├── 03-predict/
│   └── get-pheno-raster.py            predict phenology for every rice pixel
├── 04-gap-model/                      Step 4: interval (gap-RF) models
│   ├── pick-sample.jl                   sample pixels from the products
│   ├── merge-sample.py                  merge the per-province-year samples
│   ├── prepare-training-data.py         filter + extract NDVI of the samples
│   ├── train-gap-rf.py                  train the 7 single-season gap models
│   ├── pick-sample-early-late.jl        sample common early/late pixels
│   ├── merge-sample-early-late.py       merge the early-late samples
│   ├── prepare-early-late-training-data.py
│   └── train-early-late-gap-rf.py       train the early-late gap model
├── 05-correct/                        Step 5: consistency correction
│   ├── correct-pheno-raster.py          repair inverted sequences
│   └── correct-early-late-raster.py     repair the rotation interval
├── 06-postprocess/                    Step 6: post-processing
│   ├── fill-failed-pheno.py             fill failed (999) pixels
│   └── merge-country.py                 merge provinces into national rasters
├── lib/                               shared code
│   ├── common.py                        constants, file-name helpers
│   ├── raster_io.py                     GeoTiff read/write helpers
│   └── gtiffio.jl                       Julia GeoTiff read/write helpers
├── external/
│   └── SMFS-RF/                       git submodule (SMF-S implementation)
└── data/                              all input/output data (gitignored)
    ├── observation/                     station observation GPKG files
    ├── ndvi/                            the 8-day 30 m NDVI product
    ├── rice-map/                        fused rice distribution maps
    ├── sites-curve/                     extracted station NDVI curves
    ├── models/                          trained RF models
    ├── result/                          phenological rasters
    ├── sample/                          sampled pixels
    └── sample-curve/                    NDVI time series of the samples
```

## Getting started

### 1. Clone (with submodules)

```bash
git clone --recursive https://github.com/<your-name>/CCDPheno-source.git
cd CCDPheno-source
```

The [SMFS-RF](https://github.com/shenrq/SMFS-RF) repository is included as a git submodule; its nested submodule `SMF_S_Release` provides the SMF-S implementation used in Step 1. If you cloned without `--recursive`, run:

```bash
git submodule update --init --recursive
```

### 2. Dependencies

Python 3.10+ (all scripts):

- numpy, pandas, geopandas, scipy, joblib
- scikit-learn, scikit-learn-intelex (for `sklearnex`)
- rasterio, rioxarray, xarray
- pyarrow (for the feather files)
- GDAL utilities (for `gdal_merge.py`)

Julia 1.9+ (for the sampling scripts only):

- ArchGDAL, DataFrames, CSV, StatsBase, Random

### 3. Prepare the data

Put your data into the corresponding directories of `data/` (or pass
`--data-root /path/to/your/data` to every script):

| Directory        | Expected content |
|------------------|------------------|
| `data/observation/` | Station observation tables (GeoPackage), one per rice type: `single-phenology-2000-2015.gpkg`, `early-phenology-2000-2015.gpkg`, `late-phenology-2000-2015.gpkg`. Columns: `station, lat, lon, altitude, PAC, province, year` plus one column per phenological stage. |
| `data/ndvi/<province>/` | The 8-day 30 m NDVI composites (e.g. `<province>_P20WR10_<year><jday>_8Day_WGS30m_WWHD_Scale10000_V2_Uint8.tif`). Adjust `ndvi_fname()` in `lib/common.py` if your naming differs. |
| `data/rice-map/<year>/` | Fused rice distribution maps (e.g. `China-staple-crops-<province>-<year>-fusion_range-v1.tif`; `CCD-Rice-<province>-<year>-fusion_range-v1.tif` for Tianjin/Hebei before 2017; `CCD-Rice-<province>-<year>-v1.tif` for the station extraction). Pixel codes: `1|4` = single-season rice, `2` = double-season rice. |

Only the data paths need to be modified to run the pipeline; all other parameters (model hyper-parameters, versions, years, thresholds) are exposed as command-line arguments with the defaults used for the published product.

## Running the pipeline

### Step 1 — station-level data

Extract the NDVI time series of the rice pixels around each station, once per rice type:

```bash
python 01-site-curve/extract-sites-rice-NDVI.py --crop single
python 01-site-curve/extract-sites-rice-NDVI.py --crop early
python 01-site-curve/extract-sites-rice-NDVI.py --crop late
```

Estimate the SMF-S offsets (knowledge-guided pixel-level labels), once per rice type:

```bash
python 01-site-curve/site-curve-offset-SMF-S.py --crop single
python 01-site-curve/site-curve-offset-SMF-S.py --crop early
python 01-site-curve/site-curve-offset-SMF-S.py --crop late
```

### Step 2 — train the phenological-stage RF models

```bash
python 02-train-pheno-model/train-with-all-data-filter-station.py
```

This trains 24 models (3 rice types x 8 stages) and saves them into
`data/models/` with the name
`<crop>-<stage>-buffer-0.05-offset2-w8-filter0-n_estimators100-pix100-fstation2_3-all_data.joblib.z`.

### Step 3 — predict the phenological rasters

```bash
python 03-predict/get-pheno-raster.py
```

Outputs the raw products `data/result/<province>-<year>-<crop>-<stage>-v5.tif`.

### Step 4 — train the gap-RF models

Sample pixels from the corrected products (Julia required):

```bash
julia 04-gap-model/pick-sample.jl --version-out 5.4 --pickfrac 20000
julia 04-gap-model/pick-sample-early-late.jl --version 5-unified2-el-filled --pickfrac 10000
```

Merge the samples:

```bash
python 04-gap-model/merge-sample.py --version 5.4
python 04-gap-model/merge-sample-early-late.py --version 5-unified2-el-filled
```

Prepare the training data and train the models:

```bash
python 04-gap-model/prepare-training-data.py --version 5.4
python 04-gap-model/train-gap-rf.py
python 04-gap-model/prepare-early-late-training-data.py --version 5-unified2-el-filled
python 04-gap-model/train-early-late-gap-rf.py
```

### Step 5 — consistency correction

```bash
python 05-correct/correct-pheno-raster.py --in-version v5 --out-version v5-unified2
python 05-correct/correct-early-late-raster.py --in-version v5-unified2 --out-version v5-unified2-el
```

The first script repairs the inverted sequences of every rice type; the second repairs the double-season rotation interval (only the late season is shifted, so the early season is kept unchanged).

### Step 6 — post-processing

Fill the failed pixels (999) with surrounding block means:

```bash
python 06-postprocess/fill-failed-pheno.py --tag v5-unified2
python 06-postprocess/fill-failed-pheno.py --tag v5-unified2-el
```

Merge the provincial rasters into the national products:

```bash
python 06-postprocess/merge-country.py --version v5.4
```

The national products are written as
`data/result/CCDPheno-Rice-China-<crop>-<stage>-<year>-v5.4.tif`.

## Pipeline overview

```
observation GPKG ──> NDVI curves ──> SMF-S offsets ──> RF models (24)
                                                            │
rice map + NDVI ──> get-pheno-raster ──> v5 rasters         │
                                              │              │
                          pick-sample.jl ──> prepare-training-data ──> train-gap-rf
                                              │              │
                          pick-sample-early-late.jl ──> prepare-early-late ──> train-early-late
                                              │
              correct-pheno-raster ──> v5-unified2
                    │
              correct-early-late-raster ──> v5-unified2-el
                    │
              fill-failed-pheno ──> *-filled
                    │
              merge-country ──> national v5.4 products
```

## Citation

If you use this code or the CCDPheno-Rice product, please cite the paper (DOI to be added).
