from . import utilities
from . import interpret
#import scSHARP.utilities as utilities
import numpy as np
import pandas as pd
import torch
import os
from .gcn_model import GCNModel
import torch
from .pca_model import PCAModel
from sklearn.decomposition import PCA
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

class scSHARP:
    """Class that runs predictions"""

    def __init__(self, data_path, tool_preds, tool_list, marker_path, neighbors=2, config="2_40.txt", ncells="all"):
        self.data_path = data_path
        self.preds_path = tool_preds
        self.tools = tool_list
        self.marker_path = marker_path
        self.neighbors = neighbors
        self.config = config
        self.ncells = ncells
        self.pre_processed = False

        self.cell_names = None
        self.model = None
        self.final_preds = None
        self.genes = None
        self.X = None
        self.pca_obj = None
        self.keep_genes = None
        self.batch_size = None
        self.counts = None
        self.keep_cells = None
        self.confident_labels = None
        
    def run_tools(self, out_path, ref_path, ref_label_path):
        try:
            run_script = "Rscript ./rdriver.r"
            bash = run_script + " " + self.data_path + " " + out_path + " " + self.marker_path + " " + ref_path + " " + ref_label_path
            subprocess.call(bash, shell=True)
            
            self.preds_path = out_path
            
            # R output file read
        except:
            print("Something went wrong with running the R tools. ")
            
        
    def prepare_data(self, thresh):
        if os.path.exists(self.preds_path):
            all_labels = pd.read_csv(self.preds_path, index_col=0)
            if all_labels.shape[1] != len(self.tools): 
                all_labels = all_labels[self.tools]
                
        else:
            raise Exception("Prediction Dataframe not Found at " + self.data_path) # TODO should this be self.preds_path?

        # read in dataset
        if self.ncells == "all":
            self.counts = pd.read_csv(self.data_path, index_col=0)
        else:
            self.counts = pd.read_csv(self.data_path, index_col=0, nrows=self.ncells)
            all_labels = all_labels.head(self.ncells)
        self.X, self.keep_cells, self.keep_genes,self.pca_obj = utilities.preprocess(np.array(self.counts), scale=False, comps=500)
        self.genes = self.counts.columns.to_numpy()[self.keep_genes]
        #all_labels = all_labels.loc[self.keep_cells,:]

        _,self.marker_names = utilities.read_marker_file(self.marker_path)

        self.cell_names = self.marker_names.copy()
        self.cell_names.sort()

        all_labels_factored = utilities.factorize_df(all_labels, self.marker_names)
        encoded_labels = utilities.encode_predictions(all_labels_factored)

        self.confident_labels = utilities.get_consensus_labels(encoded_labels, necessary_vote = thresh)
        self.pre_processed = True
    
    def run_prediction(self, training_epochs=150, thresh=0.51, batch_size=40, seed=8):
        """Trains GCN modle on consensus labels and returns predictions"""
        self.batch_size = batch_size 
        self.prepare_data(thresh)

        train_nodes = np.where(self.confident_labels != -1)[0]
        test_nodes = np.where(self.confident_labels == -1)[0]

        dataset  = torch.utils.data.TensorDataset(torch.tensor(self.X), torch.tensor(self.confident_labels))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

        if self.model == None: self.model = GCNModel(self.config, neighbors=self.neighbors, target_types=len(self.marker_names), seed=seed)
        self.model.train(dataloader, epochs=training_epochs, verbose=True)

        preds,_ = self.model.predict(test_dataloader)
        self.conf_scores, self.final_preds = preds.max(dim=1)

        return self.final_preds, train_nodes, test_nodes, self.keep_cells, self.conf_scores

    def knn_consensus(self, k=5):
        """returns knn consensus predictions for unconfidently
        labled cells based on k nearest confident votes"""
        if not self.pre_processed:
            self.prepare_data()

        
        return utilities.knn_consensus_batch(self.X, self.confident_labels, k)

    def run_interpretation(self):
        """Runs model gradient based interpretation"""
        X,_,_,_ = utilities.preprocess(np.array(self.counts), scale=False, run_pca=False)
        pca = PCA(n_components=500, random_state=8)
        pca.fit(X)
        pca_mod = PCAModel(pca.components_, pca.mean_)
        seq = torch.nn.Sequential(pca_mod, self.model)
        #meta_path = "/home/groups/ConradLab/daniel/sharp_data/pbmc_test/labels_cd4-8.csv"
        #metadata = pd.read_csv(meta_path, index_col=0)
        #real_y = pd.factorize(metadata.iloc[:,0], sort=True)[0]
        #real_y = real_y[self.keep_cells]
        #real_y = torch.tensor(real_y)
        int_df = interpret.interpret_model(seq, X, self.final_preds, self.genes, self.batch_size, self.model.device)
        int_df.columns = self.cell_names
        
        return int_df

    def heat_map(self, att_df, out_dir=None, n=5):
        att_df = att_df.abs()
        scale_int_df = pd.DataFrame(preprocessing.scale(att_df, with_mean=False))
        scale_int_df.columns = att_df.columns
        scale_int_df.index = att_df.index
        markers = self.__get_most_expressed(scale_int_df, n)

        ax = sns.heatmap(scale_int_df.loc[markers,:])
        ax.set(xlabel="Cell Type")
        plt.plot()
        plt.show()

        if out_dir:
            plt.savefig(out_dir, format="pdf", bbox_inches="tight")
            ax.savefig(out_dir, format="pdf", bbox_inches="tight")

        return ax

    def __get_most_expressed(self, df, n=5):
        '''Get top n marker genes for each cell type'''
        markers = []
        for ctype in df.columns:
            ordered_df = df.sort_values(ctype, ascending=False).head(n)
            markers += list(ordered_df.index)

        return markers

    def save_model(self, file_path):
        """Save model as serialized object at specified path"""

        torch.save(self.model, file_path)

    def load_model(self, file_path):
        """Load model as serialized object at specified path"""

        self.model = torch.load(file_path)

    def __str__(self):
        return f'scSHARP object: Neighbors: {self.neighbors} Config path: {self.config} Num cells: {self.ncells}'
