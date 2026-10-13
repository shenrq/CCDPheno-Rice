#!/usr/bin/env julia
# -*- coding: utf-8 -*-
# AUTHOR: Shen Ruoque
#=
Sample rice pixels from the corrected phenology rasters for the gap-RF
training data. For each (province, year, rice type), a number of pixels is
randomly selected with a probability proportional to the rice area, and their
8 phenological dates are written into a CSV file.

The number of sampled pixels per province-year equals
    n_valid / pickfrac * (pixel_size / 30m)^2
so that about 1 out of `pickfrac` 30 m rice pixels is kept.

Example:
    julia 04-gap-model/pick-sample.jl --version-out 5.4 --pickfrac 20000
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
pickfrac = parse(Int, get(args, "pickfrac", "20000"))
version_out = get(args, "version-out", "5.4")
# Version tags of the corrected (and filled) provincial rasters that are
# sampled. Single-season rice may use a different tag than early/late rice.
tag_single = get(args, "tag-single", "5-unified2-filled")
tag_double = get(args, "tag-double", "5-unified2-el-filled")
inpath = get(args, "in-dir", joinpath(@__DIR__, "..", "data", "result"))
sample_path = get(args, "sample-dir", joinpath(@__DIR__, "..", "data", "sample", "province"))
mkdir(sample_path; recursive=true)

places = [
    "Jiangsu", "Sichuan", "Heilongjiang", "Yunnan", "Chongqing",
    "Guizhou", "Jilin", "Liaoning", "Henan", "Shanghai", "Shaanxi",
    "Shandong", "Hebei", "Inner_Mongolia", "Ningxia", "Tianjin",
    "Hunan", "Jiangxi", "Hubei", "Guangxi", "Anhui", "Zhejiang",
    "Fujian", "Guangdong", "Hainan",
]

phenos = [
    "transplanting", "regreening", "tillering", "stem_elongation",
    "booting", "heading", "milk_ripening", "maturity",
]

function get_version_in(ricetype)
    ricetype == "single" ? tag_single : tag_double
end

function picksample(infile)
    phenology, ref = readgtiff(infile)
    geotrans = ref[:geotransform]
    tot_cover_id = findall(phenology[:, :, 1] .!= -999)
    pixnum = round(Int, length(tot_cover_id) / pickfrac * (geotrans[2] / res30) ^ 2)
    cover_id = sample(tot_cover_id, pixnum; replace=false)
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
    if place in ["Guangdong", "Hainan"]
        ricetypes = ["early", "late"]
    elseif place in ["Hunan", "Jiangxi", "Hubei", "Guangxi", "Anhui", "Zhejiang", "Fujian"]
        ricetypes = ["single", "early", "late"]
    else
        ricetypes = ["single"]
    end

    yrs = place in ["Hebei", "Tianjin"] ? (2001:2016) : (2001:2024)

    for ricetype = ricetypes
        @threads for yr = yrs
            version_in = get_version_in(ricetype)
            outfile = joinpath(sample_path, "sample-$place-$yr-$ricetype-v$version_out.csv")
            if isfile(outfile)
                continue
            end
            sheet = DataFrame()
            infile0 = joinpath(inpath, "$place-$yr-$ricetype-transplanting-v$version_in.tif")
            cover_id, coords = picksample(infile0)
            sheet[!, :longitude] = coords[1]
            sheet[!, :latitude] = coords[2]
            for pheno = phenos
                infile = joinpath(inpath, "$place-$yr-$ricetype-$pheno-v$version_in.tif")
                phenolist = sample_tags(infile, cover_id)
                sheet[!, Symbol(pheno)] = phenolist
            end
            CSV.write(outfile, sheet; delim=",")
        end
    end
end
