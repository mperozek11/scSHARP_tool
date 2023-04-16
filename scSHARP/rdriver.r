# This script connects our python package to our R package

# loading our R package
library(R4scSHARP)

# read command line arguments
args = commandArgs(trailingOnly=TRUE)

# get necessary paths from python package
data_path <- args[1]
out_path <- args[2]
marker_path <- args[3]
ref_path <- args[4]
ref_label_path <- args[5]
tools <- args[6]

print(out_path)

# run R tools
output <- run_r4scsharp(data_path, marker_path,
    ref_path, ref_label_path, out_path=out_path, tools=tools)
