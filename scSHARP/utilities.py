import torch
from torch_cluster import knn_graph
from torch.nn import Sequential as Seq, Linear, SiLU, Dropout, ELU, Tanh
import anndata as ad
import scanpy as sc
from sklearn.decomposition import PCA
import numpy as np
import random
import math
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import json
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from captum.attr import IntegratedGradients, DeepLift, DeepLiftShap, FeaturePermutation
import math
from sklearn.metrics import confusion_matrix
from collections import Counter
from os.path import exists
import pkg_resources
from . import gcn_model

"""General functions and definitions"""

def preprocess(data, normalize=True, scale=False, targetsum=1e4, run_pca=True, comps=500, cell_fil=0, gene_fil=0):
    """
    Preprocesses raw counts DGE matrix

    The default parameter values assume filtered, but not normalized DGE counts matrix 
    with rows representing cells and columns representing genes

    Parameters
    ----------
    normalize: bool
        row norm and lognorm
    scale: bool
        scale by gene to mean 0 and std 1 
    targetsum: float
        row norm then multiply by target sum
    run_pca: bool
        Whether or not to run PCA
    comps: int
        how many components to use for PCA
    cel_fil: int
        Filter param. Minimum number of cells containing a given gene to be included
    gene_fil: int
        Filter param. Minimum number of genes containing a given cell to be included

    Returns
    -------
    preprocessed dataset as an nD-array
    """

    adata = ad.AnnData(data, dtype=data.dtype)
    row_filter, _ = sc.pp.filter_cells(adata, min_genes=cell_fil, inplace=False)
    col_filter,_ = sc.pp.filter_genes(adata, min_cells=gene_fil, inplace=False)
    subset = adata[row_filter,col_filter]
    adata = ad.AnnData(subset.X, dtype=subset.X.dtype)
    if normalize:
        sc.pp.normalize_total(adata, target_sum=targetsum, layer=None)
    new_data = adata.X
    if normalize:
        new_data = sc.pp.log1p(new_data)
    
    if scale:
        new_data = sc.pp.scale(new_data)
    non_pca_data = new_data.copy()
    pca = None
    if run_pca:
        pca = PCA(n_components=comps, random_state=8)
        new_data = pca.fit_transform(new_data)

    return new_data, row_filter, col_filter, pca, non_pca_data

def mask_labels(labels, masking_pct):
    """masks labels for training

    Randomly masks a specified portion of the labels, substituting their value for -1

    Parameters
    ----------
    labels: list of labels
    masking_pct: float value for proportion of masked rows

    Returns
    -------
    Tuple of:
        labels: original list of labels 
        masked_labels: copy of original labels, with masking applied
    """
    random.seed(8)
    subset = random.sample(range(len(labels)), math.floor((len(labels)*masking_pct)))
    masked_labels = np.copy(labels)
    masked_labels[subset] = -1 

    return labels, masked_labels  

def read_marker_file(file_path):
    """parses marker file
    
    Returns
    -------
    Tuple of:
        markers: list of marker genes 
        marker_names: list of string gene names
    """

    marker_df = pd.read_csv(file_path, header=None, index_col=0)
    markers = np.array(marker_df).tolist()
    marker_names = list(marker_df.index)
    """with open(file_path) as f:
        lines = f.readlines()
        f.close()
    
    markers = [[] for _ in range(len(lines))]
    marker_names = [""]*len(markers)
    for i, line in enumerate(lines):
        temp_name = line.split(sep=":")[0]
        temp_markers = line.split(sep=":")[1].split(sep=",")
        temp_markers = [s.strip() for s in temp_markers]
        markers[i] = temp_markers
        marker_names[i] = temp_name"""

    return markers, marker_names

def label_counts(data_path, tools, ref_path, ref_label_path, marker_path):
    """Label inputted dataset with mutlitple annotation tools"""

    markers, marker_names = read_marker_file(marker_path)

    ro.r.source('tools/r_tools.R')
    #markers = 
    preds = ro.r.run(data_path, tools, markers, marker_names, ref_path, ref_label_path)
    with localconverter(ro.default_converter + pandas2ri.converter):
        preds = ro.conversion.rpy2py(preds)
    
    return preds

def load_model(file_path, target_types):
        """loads model from json format"""
        
        if not exists(file_path):
            f = pkg_resources.resource_stream(__name__, file_path)
        else:
            f = open(file_path)
        data = json.load(f)
        f.close()
        final_layers = []

        for layer in data['layers']:
            if layer['type'] == 'EdgeConv':
                new_layer = gcn_model.EdgeConv(layer['input'],layer['output'])
            
            elif layer['type'] == 'Linear':
                new_layer = torch.nn.Linear(layer['input'], layer['output'])
            
            elif layer['type'] == 'Sigmoid':
                new_layer = torch.nn.Sigmoid()
            
            elif layer['type'] == 'BatchNorm':
                new_layer = torch.nn.BatchNorm1d(layer['input'])

            else:
                raise Exception("Unrecognizable layer type:" + layer['type'])
            
            final_output = layer['output']
            final_layers.append(new_layer)
        
        final_layers.append(torch.nn.Linear(final_output,target_types))
        return final_layers

def factorize_df(df, all_cells):
    """factorizes all columns in pandas df"""

    # add array with all cell options so factor is the same for each column
    dummy = []
    for i in range(len(all_cells)):
        dummy.append([all_cells[i]]*df.shape[1])
    dummy = pd.DataFrame(dummy)
    dummy.columns = df.columns
    df = pd.concat([df,dummy])
    
    temp = df.apply(pd.factorize, axis=0, sort=True)
    keys = temp.iloc[1,0]
    temp = temp.iloc[0,:]
    indices = list(temp.index)
    d = {key: None for key in indices}
    for i in range(temp.shape[0]):
        d[indices[i]] = temp.iloc[i]

    return pd.DataFrame(d).iloc[:-len(all_cells)], keys

def encode_predictions(df):
    """encodes predictions for each cell with 1 for each prediction"""
    all_preds = []
    for i in range(df.shape[1]):
        all_preds.append(df.iloc[:,i].to_numpy())
    all_preds = np.array(all_preds).flatten()
    
    #add -1 then remove so encoder takes into account unknowns even if there isn't any
    all_preds = np.append(all_preds, -1)
    enc = OneHotEncoder(drop='first')
    encoded_y = enc.fit_transform(all_preds.reshape(-1,1)).toarray()
    encoded_y = encoded_y[:-1,:]
    # need to add three scores together
    final_encoded = np.zeros(shape=(df.shape[0],encoded_y.shape[1]))
    scoring_length = df.shape[0]
    lower =0
    upper = scoring_length
    for i in range(int(len(encoded_y)/df.shape[0])):
        final_encoded += encoded_y[lower:upper,:]
        lower = upper
        upper += scoring_length
    
    return final_encoded

def pred_accuracy(preds, real):
    """returns accuracy of predictions"""
    return float((torch.tensor(preds) == torch.tensor(real)).type(torch.FloatTensor).mean().numpy())

def get_consensus_labels(encoded_y, necessary_vote):
    """method that gets consensus vote of multiple prediction tools
    If vote is < 1 then taken as threshold pct to be >= to
    """
    confident_labels = np.zeros(shape = (encoded_y.shape[0],))
    for i in range(encoded_y.shape[0]):
        row = encoded_y[i,:]

        if necessary_vote < 1:
            if row.sum() != 0: 
                row = row / row.sum()
        
        max_index = np.argmax(row)
        if row[max_index] >= (necessary_vote):
            confident_labels[i] = max_index
        else: confident_labels[i] = -1
    
    return confident_labels

def filter_scores(scores, thresh = 0.5):
    """filters out score columns with NAs > threshold"""
    keep_cols = []
    for col in scores.columns:
        pct_na = (scores[col].isna().sum()) / scores.shape[0]

        if pct_na < thresh: keep_cols.append(col)
    
    return scores[keep_cols]


def weighted_encode(df, encoded_y, tool_weights):
    """More advanced consensus method
    df: cells x tools
    tool_weights: cell_types x tools
    """
    final_encode = np.zeros(encoded_y.shape)
    votes = df.to_numpy()
    for i in range(encoded_y.shape[0]):
        votes_row = votes[i,:]
        row = encoded_y[i,:]
        row = row / row.sum()
        max_index = np.argmax(row)
        
        if np.argwhere(row == row[max_index]).flatten().shape[0] > 1:
            #do something
            #maybe weight be average in this case?
            print('multiple maxes')
            continue
        
        # max index is winning cell type
        # for that winner, use weights of tools for it
        weights = tool_weights[max_index]
        weighted_encode = np.zeros(encoded_y.shape[1])
        for j in range(votes_row.shape[0]):
            encode = np.zeros(encoded_y.shape[1])
            encode[votes_row[j]] = 1
            encode = encode * weights[j]
            weighted_encode += encode
        
        final_encode[i,:] = weighted_encode
    
    return final_encode

def validation_metrics(real_y, preds, train_nodes, test_nodes):
    all_equality = (real_y == preds)
    all_accuracy = all_equality.type(torch.FloatTensor).mean()

    all_cm = confusion_matrix(real_y, preds)

    train_equality = (real_y[train_nodes] == preds[train_nodes])
    train_accuracy = train_equality.type(torch.FloatTensor).mean()

    train_cm = confusion_matrix(real_y[train_nodes], preds[train_nodes])

    test_equality = (real_y[test_nodes] == preds[test_nodes])
    test_accuracy = test_equality.type(torch.FloatTensor).mean()

    test_cm = confusion_matrix(real_y[test_nodes], preds[test_nodes])

    return float(all_accuracy), all_cm, float(train_accuracy), train_cm, float(test_accuracy), test_cm

def get_max_consensus(votes):
    """Gets max consensus"""
    final_preds = []
    for row in votes:
        max_index = np.argmax(row)
        if np.count_nonzero(row == row[max_index]) > 1:
            final_pred = -1
        else:
            final_pred = max_index

        final_preds.append(final_pred)
    return torch.tensor(final_preds)

def knn_consensus(counts, preds, n_neighbors, converge=False, one_epoch=False):
    """Do kNN consensus, iterate until x% do not change"""

    knn = knn_graph(torch.FloatTensor(counts), k=n_neighbors)
    preds = np.array(preds)
    count = 0
    while True:
        new_preds = preds.copy()
        changed = 0
        for i in range(counts.shape[0]):
            sub_knn = knn[:,knn[1,:]==i]
            neighbors = sub_knn[0,:]
            neighbor_preds = preds[neighbors]
            if type(neighbor_preds) == np.int64 or type(neighbor_preds) == np.float64: neighbor_preds = np.array([neighbor_preds])
            vote_counts = Counter(neighbor_preds).most_common()
            if len(vote_counts) == 1:
                if vote_counts[0][0] != -1:
                    # update preds
                    new_preds[i] = vote_counts[0][0]
            
            elif vote_counts[0][1] != vote_counts[1][1] and vote_counts[0][0] != -1:
                # update preds
                new_preds[i] = vote_counts[0][0]

        #if np.mean( preds != new_preds ) < .05: break
        if len(new_preds[new_preds == -1]) == 0 and converge==False:
            preds = new_preds
            break
        if np.mean( preds != new_preds ) < .05 and converge==True: 
            preds = new_preds
            break
        if one_epoch == True:
            preds = new_preds
            break
        
        preds = new_preds
        count += 1
        print(count)
        if count > 50: break
        #break
    return preds

def knn_consensus_batch(counts, preds, n_neighbors, converge=False, one_epoch=False, batch_size=1000, keep_conf=False):
    """Do kNN consensus, iterate until x% do not change"""

    preds = np.array(preds)
    count = 0
    while True:
        new_preds = torch.tensor([])
        dataset  = torch.utils.data.TensorDataset(torch.tensor(counts), torch.tensor(preds))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for batch, labels in dataloader:
            knn = knn_graph(batch, k=n_neighbors)
            loc_new_preds = torch.clone(labels)
            for i in range(batch.shape[0]):
                if keep_conf==True and labels[i] != -1: continue
                sub_knn = knn[:,knn[1,:]==i]
                neighbors = sub_knn[0,:]
                neighbor_preds = labels[neighbors]
                if type(neighbor_preds) == np.int64 or type(neighbor_preds) == np.float64: neighbor_preds = np.array([neighbor_preds])
                vote_counts = Counter(neighbor_preds.detach().numpy()).most_common()
                if len(vote_counts) == 1:
                    if vote_counts[0][0] != -1:
                        # update preds
                        loc_new_preds[i] = float(vote_counts[0][0])

                elif vote_counts[0][1] != vote_counts[1][1] and vote_counts[0][0] != -1:
                    # update preds
                    loc_new_preds[i] = float(vote_counts[0][0])
            new_preds = torch.cat((new_preds,loc_new_preds))
        #if np.mean( preds != new_preds ) < .05: break
        if len(new_preds[new_preds == -1]) == 0 and converge==False:
            preds = new_preds
            break
        if np.mean( np.array(preds) != np.array(new_preds) ) < .05 and converge==True:
            preds = new_preds
            break
        if one_epoch == True: 
            preds = new_preds
            break
        preds = new_preds
        count += 1
        if count % 10 == 0:
            print(f'kNN iter: {count}')
        if count > 50: break
        #break
    return preds.detach().numpy()
