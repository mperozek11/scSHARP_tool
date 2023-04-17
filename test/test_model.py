import numpy as np
import pandas as pd
import torch
from gcn_model import GCNModel
import utilities
import statistics
import os

def test_model(data_folders, tool_list, votes, model_file, neighbors, nbatch, training_epochs, random_inits, markers="markers.txt", counts="query_counts.csv", meta="query_meta.csv", meta_col = "Group", preds ="preds.csv"):
    
    final_df = pd.DataFrame({"data_name":[], "method":[], "total_accuracy":[],"train_accuracy":[], "test_accuracy":[], "total_sd":[], "train_sd":[], "test_sd":[]})
    # for all datasets
    for data_folder in data_folders:
        # get labels
        data_path = data_folder + counts
        tools = tool_list
        ref_path = data_folder + "ref_counts.csv"
        ref_label_path = data_folder + "ref_labels.csv"
        marker_path = data_folder + markers
        if os.path.exists(data_folder + preds):
            all_labels = pd.read_csv(data_folder + preds, index_col=0)
            if all_labels.shape[1] != len(tools): 
                all_labels = all_labels[tools]
                #raise Exception("wrong amount of tools in file")
        else:
            raise Exception('generate predictions first')
            # all_labels = utilities.label_counts(data_path,tools,ref_path,ref_label_path,marker_path)
        
        # read in dataset
        X = pd.read_csv(data_path, index_col=0)
        X, keep_cells,_,_ = utilities.preprocess(np.array(X), scale=False)

        #all_labels = all_labels.loc[keep_cells,:]

        _,marker_names = utilities.read_marker_file(marker_path)

        all_labels_factored = utilities.factorize_df(all_labels, marker_names)
        encoded_labels = utilities.encode_predictions(all_labels_factored)

        meta_path = data_folder + meta
        metadata = pd.read_csv(meta_path, index_col=0)
        if type(meta_col)==int:
            real_y = pd.factorize(metadata.iloc[:,meta_col], sort=True)[0]
        else: real_y = pd.factorize(metadata[meta_col], sort=True)[0]
        real_y = real_y[keep_cells]

        confident_labels = utilities.get_consensus_labels(encoded_labels, necessary_vote = votes)
        train_nodes = np.where(confident_labels != -1)[0]
        test_nodes = np.where(confident_labels == -1)[0]
        
        dataset  = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(confident_labels))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=nbatch, shuffle=True)

        test_dataset  = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(real_y))
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=nbatch, shuffle=False)
        
        train_accuracies = [0]*random_inits
        test_accuracies = [0]*random_inits
        total_accuracies = [0]*random_inits

        for i in range(random_inits):
            m = GCNModel(model_file, neighbors, target_types=len(marker_names), seed=i)
            m.train(dataloader, training_epochs, verbose=True)
            metrics = m.validation_metrics(test_dataloader, train_nodes, test_nodes)
            total_accuracies[i] = metrics[0]
            train_accuracies[i] = metrics[2]
            test_accuracies[i] = metrics[4]

        print(total_accuracies)
        # max of columns pred.
        #max_pred = torch.tensor(encoded_labels).max(dim=1)[1]
        max_pred = utilities.get_max_consensus(encoded_labels)

        dataset_names = ["GCN", "Max Col.", "Confident Labels"] + tool_list + ["Tool Avg."]
        # full dataset accuracy
        all = []
        all.append(statistics.mean(total_accuracies))
        all.append(utilities.pred_accuracy(max_pred, real_y))
        all.append(None)
        for tool in tool_list:
            all.append(utilities.pred_accuracy(np.array(all_labels_factored[tool]), real_y))
        all.append(statistics.mean(all[3:]))

        # train set accuracy
        all_trains = []
        all_trains.append(statistics.mean(train_accuracies))
        all_trains.append(utilities.pred_accuracy(max_pred[train_nodes], real_y[train_nodes]))
        all_trains.append(utilities.pred_accuracy(confident_labels[train_nodes], real_y[train_nodes]))
        for tool in tool_list:
            all_trains.append(utilities.pred_accuracy(np.array(all_labels_factored[tool][train_nodes]), real_y[train_nodes]))
        all_trains.append(statistics.mean(all_trains[3:])) 

        # test set accuracy
        all_tests = []
        all_tests.append(statistics.mean(test_accuracies))
        all_tests.append(utilities.pred_accuracy(max_pred[test_nodes], real_y[test_nodes]))
        all_tests.append(None)
        for tool in tool_list:
            all_tests.append(utilities.pred_accuracy(np.array(all_labels_factored[tool][test_nodes]), real_y[test_nodes]))
        all_tests.append(statistics.mean(all_tests[3:]))        

        # sds
        all_sds = [0]*len(dataset_names)
        train_sds = [0]*len(dataset_names)
        test_sds = [0]*len(dataset_names)

        all_sds[0] = statistics.stdev(total_accuracies)
        train_sds[0] = statistics.stdev(train_accuracies)
        test_sds[0] = statistics.stdev(test_accuracies)

        all_sds[-1] = statistics.stdev(all[3:-1])
        train_sds[-1] = statistics.stdev(all_trains[3:-1])
        test_sds[-1] = statistics.stdev(all_tests[3:-1])

        data_name = [data_folder.split("/")[-2]]*len(dataset_names)
        df = pd.DataFrame({"data_name":data_name, "method":dataset_names, "total_accuracy":all,"train_accuracy":all_trains, "test_accuracy":all_tests, "total_sd":all_sds, "train_sd":train_sds, "test_sd":test_sds})
        final_df = pd.concat([final_df, df])

    return final_df

if __name__ == "__main__":
    
    data_folders = ["/home/groups/ConradLab/daniel/sharp_data/pbmc_test/"]
    tools = ["sctype","scsorter","scina", "singler", "scpred"]
    #tools = ["sctype","scsorter","scina"]
    votes_necessary = .51
    model_file = "configs/2_25.txt"
    neighbors = 2
    batch_size=50
    training_epochs=150
    random_inits = 5
    counts="counts.csv"
    meta="labels_cd4-8.csv"
    meta_col = 0
    df = test_model(data_folders, tools, votes_necessary, model_file, neighbors, batch_size, training_epochs, random_inits, counts=counts, meta=meta, meta_col=meta_col)
    df.to_csv("pbmc_test_results.csv")
