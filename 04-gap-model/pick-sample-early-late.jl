#!/usr/bin/env julia
# -*- coding: utf-8 -*-
# AUTHOR: Shen Ruoque
#=
Sample pixels from the early- and late-season rice phenology rasters
simultaneously, so that the early-maturity and late-transplanting dates are
sampled from the same pixels (double-season rice is grown on the same field).
These samples are used to train the early-late gap-RF model.

Example:
    julia 04-gap-model/pick-sample-early-late.jl --version 5-unified2-el-filled --pickfrac 10000
=#
using Base.Threads
using Printf
import ArchGDAL as AG
using DataFrames, CSV
include(joinpath(@__DIR__, "..", "lib", "gtiffio.jl"))
using Random
using StatsBase

# --- command line arguments -----------------------------------------------
function parse_args()
    args = Dict{String,String}()
    i = 1
    while i <= length(ARGS)
        if startswith(ARGS[i], "--")
            key = ARGS[i][3:end]
            if i + 1 <= length(ARGS)
                args[key] = ARGS[i + 1]
                i += 2
            else
                i += 1
            end
        else
            i += 1
        end
    end
    args
end

args = parse_args()

res30 = 0.000269494585236
pickfrac = parse(Int, get(args, "pickfrac", "10000"))
version = get(args, "version", "5-unified2-el-filled")
inpath = get(args, "in-dir", joinpath(@__DIR__, "..", "data", "result"))
sample_path = get(args, "sample-dir", joinpath(@__DIR__, "..", "data", "sample", "early-late"))
mkdir(sample_path; recursive=true)

places = [
    "Hunan", "Jiangxi", "Hubei", "Guangxi", "Anhui",
    "Zhejiang", "Fujian", "Guangdong", "Hainan",
]

function picksample_common(infile_early, infile_late)
    early_maturity, ref = readgtiff(infile_early)
    late_transplanting, _ = readgtiff(infile_late)
    geotrans = ref[:geotransform]

    # Pixels where both the early maturity and the late transplanting are valid.
    common = findall(
        (early_maturity[:, :, 1] .!= -999) .&
        (late_transplanting[:, :, 1] .!= -999)
    )
    if isempty(common)
        return nothing
    end
    pixnum = round(Int, length(common) / pickfrac * (geotrans[2] / res30) ^ 2)
    pixnum = min(pixnum, length(common))
    cover_id = sample(common, pixnum; replace=false)
    pick_coord = [
        [geotrans[1] + (i[1] - 1 + 0.5) * geotrans[2] for i = cover_id],
        [geotrans[4] + (i[2] - 1 + 0.5) * geotrans[6] for i = cover_id],
    ]
    cover_id, pick_coord
end

function sample_tags(infile, cover_id)
    phenology, _ = readgtiff(infile)
    phenology[cover_id]
end

for place = places
    println(place)
    for yr = 2001:2024
        outfile = joinpath(sample_path, "sample-$place-$yr-v$version.csv")
        if isfile(outfile)
            continue
        end

        infile_early = joinpath(inpath, "$place-$yr-early-maturity-v$version.tif")
        infile_late = joinpath(inpath, "$place-$yr-late-transplanting-v$version.tif")
        if !isfile(infile_early) || !isfile(infile_late)
            continue
        end

        sampled = picksample_common(infile_early, infile_late)
        if sampled === nothing
            println("  $place $yr: no common pixels")
            continue
        end
        cover_id, coords = sampled

        sheet = DataFrame()
        sheet[!, :longitude] = coords[1]
        sheet[!, :latitude] = coords[2]
        sheet[!, :early_maturity] = sample_tags(infile_early, cover_id)
        sheet[!, :late_transplanting] = sample_tags(infile_late, cover_id)
        sheet[!, :gap] = sheet.late_transplanting .- sheet.early_maturity
        CSV.write(outfile, sheet; delim=",")
    end
end
