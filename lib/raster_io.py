#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared raster I/O helpers used by the CCDPheno-Rice production scripts.

The functions emulate a MATLAB-like workflow: read a GeoTiff into a numpy
array together with its georeference, then write results back on the same
grid.
"""
import numpy as np
import rasterio


def gtiffref(file):
    """Return the georeference of a GeoTiff file as a dict."""
    with rasterio.open(file) as ds:
        ref = {
            "width": ds.width,
            "height": ds.height,
            "count": ds.count,
            "dtype": ds.dtypes[0],
            "transform": ds.transform,
            "crs": ds.crs,
        }
    return ref


def readgtiff(file):
    """Read a GeoTiff file. Return (data array, reference)."""
    with rasterio.open(file) as ds:
        data = ds.read()
    return data, gtiffref(file)


def writegtiff(file, data, ref0, compress="DEFLATE", bigtiff=False, nodata=None, nthread=None):
    """Write a numpy array to a GeoTiff file following the given reference."""
    ref = ref0.copy()
    if data.ndim == 2:
        data = data[np.newaxis, :, :]
    ref["count"] = data.shape[0]
    ref["dtype"] = data.dtype
    if compress is not None:
        ref["compress"] = compress
        ref["tiled"] = True
    if bigtiff:
        ref["BIGTIFF"] = "YES"
    if nodata is not None:
        ref["nodata"] = nodata
    if nthread is not None:
        ref["NUM_THREADS"] = nthread
    with rasterio.open(file, "w", driver="GTiff", **ref) as ds:
        ds.write(data)
    return file
