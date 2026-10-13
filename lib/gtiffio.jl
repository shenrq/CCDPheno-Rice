#!/usr/bin/env julia
# -*- coding: utf-8 -*-
# AUTHOR  : Shen Ruoque
#=
gtiffio.jl
MATLAB-like helpers to read and write GeoTiff files.
Tested with julia v1.9.0 and ArchGDAL v0.9.3.
=#
import ArchGDAL as AG

"""
    gtiffref(file)

Read the georeference of a GeoTiff file and return it as a Dict.
"""
function gtiffref(file :: String)
    AG.read(file) do ds
        Dict(
            :width => AG.width(ds),
            :height => AG.height(ds),
            :nbands => AG.nraster(ds),
            :dtype => AG.pixeltype(ds),
            :geotransform => AG.getgeotransform(ds),
            :proj => AG.getproj(ds)
        )
    end
end

"""
    readgtiff(file)

Read a GeoTiff file. Return a 3-D data array and its reference.
"""
function readgtiff(file :: String)
    AG.read(file) do ds
        AG.read(ds)
    end, gtiffref(file)
end

"""
    readgtiff(file, band)

Read a single band of a GeoTiff file. Return a 2-D band array and its reference.
"""
function readgtiff(file :: String, band :: Integer)
    AG.read(file) do ds
        AG.getband(ds, band) do b
            AG.read(b)
        end
    end, gtiffref(file)
end

"""
    writegtiff(file, data, ref; compress=true, ctype="DEFLATE", bigtiff=false, nthread=nothing)

Write a numpy-array-like data into a GeoTiff file following the given reference.
"""
function writegtiff end

function writegtiff(
    file :: String, data :: AbstractArray{T, 3}, ref :: Dict;
    compress="DEFLATE", bigtiff=false, nodata=nothing, nthread=nothing
) where T
    options = []
    if !isnothing(compress) append!(options, ["COMPRESS=$compress", "TILED=YES"]) end
    if bigtiff append!(options, ["BIGTIFF=YES"]) end
    if !isnothing(nthread) append!(options, ["NUM_THREADS=$nthread"]) end
    AG.create(
        file; driver=AG.getdriver("GTiff"),
        width=ref[:width], height=ref[:height], nbands=size(data, 3),
        dtype=T, options=options
    ) do ds
        AG.setgeotransform!(ds, ref[:geotransform])
        AG.setproj!(ds, ref[:proj])
        if !isnothing(nodata)
            for band = 1:size(data, 3)
                rb = AG.getband(ds, band)
                AG.setnodatavalue!(rb, nodata)
            end
        end
        AG.write!(ds, data, Int32.(1:size(data, 3)))
    end
    file
end

writegtiff(file, data :: AbstractArray{<: Real, 2}, ref; kargs...) = writegtiff(
    file, reshape(data, size(data)..., :), ref; kargs...
)
