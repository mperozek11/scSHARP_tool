library(R4scSHARP)

args = commandArgs(trailingOnly=TRUE)

# input vars:
# paths should be relative to your working directory
# (call getwd() to get this information).

data_path <- args[1]
out_path <- args[2]
marker_path <- args[3]
ref_path <- args[4]
ref_label_path <- args[5]

print(out_path)


output <- run_r4scsharp(data_path, out_path, marker_path,
    ref_path, ref_label_path)