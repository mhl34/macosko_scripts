#!/usr/bin/env Rscript
args = commandArgs(trailingOnly = T)
ref_obj_path <- args[1]
query_obj_path <- args[2]
ref_obj_filename <- strsplit(df, split = "/")