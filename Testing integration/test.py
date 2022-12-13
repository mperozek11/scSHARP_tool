import sys
sys.path.append('../')
from scSHARP.sc_sharp import scSHARP

data_path = "../simulations/splat_0.7/query_counts.csv.gz"
tool_preds = "../simulations/splat_0.7/preds.csv"
tool_list = ["scina", "scsorter", "sctype", "scpred", "singler"]
out_path = "."
marker_path = "../simulations/splat_0.7/markers.txt"
neighbors=2
config="../configs/2_25.txt"
ref_path = "./splat_0.8/ref_counts.csv.gz"
ref_label_path = "./splat_0.8/ref_labels.csv"

sharp = scSHARP(data_path = data_path, tool_preds = tool_preds, tool_list = tool_list, marker_path = marker_path, neighbors = neighbors, config = config)

sharp.run_tools(out_path, ref_path, ref_label_path)

preds, train_nodes, test_nodes, keep_cells, conf_scores = sharp.run_prediction(training_epochs=30, thresh=0.51, batch_size=20, seed=8)