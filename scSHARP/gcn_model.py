from . import utilities
import torch
from torch.nn import Sequential as Seq, Linear, SiLU, Dropout, ELU, Tanh
from torch_cluster import knn_graph
from sklearn.metrics import confusion_matrix
from torch_geometric.nn import MessagePassing



class GCNModel(torch.nn.Module):
    """class for label propagation GCN model"""

    def __init__(self, config_file, neighbors, target_types, seed=8, dropout=0.0):
        super().__init__()
        torch.manual_seed(seed)
        self.config = config_file
        self.neighbors = neighbors
        self.dropout = dropout
        self.layers = utilities.load_model(config_file, target_types)
        self.opt = torch.optim.Adam([weight for item in self.layers for weight in list(item.parameters())],0.0001)
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")

    def forward(self, X, edge_index=None):
        """forward pass through model"""

        dropout = torch.nn.Dropout(p=self.dropout)
        X = dropout(X)
        for layer in self.layers:
            #if on linear layer, no need to construct graph
            if type(layer) == EdgeConv:
                if edge_index==None:
                    edge_index = self.construct_graph(X)
                X = layer(X, edge_index)                
            else:
                X = layer(X)

        return X
    
    def train(self, trainloader, epochs, verbose=True, ignore_index_input = -1):
        """train model"""
        self.to_device()
        for epoch in range(epochs):
            epoch_loss = 0

            for local_batch, local_label in trainloader:

                local_batch, local_label = local_batch.to(self.device), local_label.to(self.device)

                self.opt.zero_grad()

                # feed data through the model - "forward pass"
                current = local_batch.float()
                current = self.forward(current)
                
                Yt = local_label.long()
                # get loss value 
                loss_func = torch.nn.CrossEntropyLoss(ignore_index=ignore_index_input)
                loss = loss_func(current, Yt) # in your training for-loop    

                epoch_loss += loss
                loss.backward()
                self.opt.step()
            
            if verbose and epoch % 10 == 0:
                print("Loss in epoch %d = %f" % (epoch, epoch_loss))

    def predict(self, testloader):
        """predict with model"""
        self.to_device()
        all_preds = []
        true_preds = []

        for local_batch, local_label in testloader:

            local_batch, local_label = local_batch.to(self.device), local_label.to(self.device)

            # feed data through the model - "forward pass"
            current = local_batch.float()
            current = self.forward(current)
            all_preds.append(current)
            true_preds.append(local_label.long())
        
        combined = torch.cat(all_preds, dim=0)
        softmax = torch.nn.Softmax(dim=1)
        combined = softmax(combined)

        real_y = torch.cat(true_preds, dim=0)

        return combined, real_y

    def validation_metrics(self, testloader, train_nodes, test_nodes):
        """returns validation metrics (ROC, prC, accuracy, etc)
            test_nodes is array with indices of nodes whose labels were not used for training
        """
        preds, real_y = self.predict(testloader)
        
        final_pred = preds.max(dim=1)[1]
        real_y = real_y.cpu()
        final_pred = final_pred.cpu()
        
        all_equality = (real_y == final_pred)
        all_accuracy = all_equality.type(torch.FloatTensor).mean()

        all_cm = confusion_matrix(real_y, final_pred)

        train_equality = (real_y[train_nodes] == final_pred[train_nodes])
        train_accuracy = train_equality.type(torch.FloatTensor).mean()

        train_cm = confusion_matrix(real_y[train_nodes], final_pred[train_nodes])

        test_equality = (real_y[test_nodes] == final_pred[test_nodes])
        test_accuracy = test_equality.type(torch.FloatTensor).mean()

        test_cm = confusion_matrix(real_y[test_nodes], final_pred[test_nodes])

        return float(all_accuracy), all_cm, float(train_accuracy), train_cm, float(test_accuracy), test_cm

    def construct_graph(self, X):
        """construct graph from data"""
        
        return knn_graph(X, k=self.neighbors, loop=True)

    def to_device(self):
    #    for layer in self.layers:
    #        layer.to(self.device)
        self.to(self.device)
    

    def to(self, device):
        """send layers to device"""
        for layer in self.layers:
            layer.to(self.device)



class EdgeConv(MessagePassing):
    """Edge convolutional layer definition"""

    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add') #  "Max" aggregation.
        self.mlp = Seq(Linear(2 * in_channels, out_channels),
                       SiLU(),
                       Linear(out_channels, out_channels))

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_i has shape [E, in_channels]
        tmp = torch.cat([x_i, x_j - x_i], dim=1)  # tmp has shape [E, 2 * in_channels]
        return self.mlp(tmp)
