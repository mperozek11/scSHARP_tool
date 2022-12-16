import torch
import pandas as pd
import numpy as np
from captum.attr import IntegratedGradients, DeepLift, DeepLiftShap, FeaturePermutation

def interpret_model(model, X, predictions, genes, batch_size, device, batches=None):
    """Performs PCA interpretation on model

    Parameters
    ----------
    model:
    X:
    predictions: 
    genes: gene names for output dataframe labels
    batch_size: size of batch for deeplift interpretation
    device: torch device for running deeplift computations
    batches: Number of batches to run dataset on deeplift. If None, run deeplift on entire dataset (default: None)
    """
    dataset  = torch.utils.data.TensorDataset(torch.FloatTensor(X), predictions.cpu())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predictions = predictions.cpu()
    prediction_names = predictions.unique().tolist()
    classes = [None]*len(prediction_names)
    for i, pred in enumerate(prediction_names):
        classes[i] = np.where(predictions == pred)[0]

    dl = DeepLift(model)
    #dl = FeaturePermutation(model)
    #attributions = np.zeros((X.shape[0], X.shape[1]))
    temp_atts = None
    counter = 0
    for data, preds in dataloader:
        baseline = torch.FloatTensor(np.zeros(data.shape))
        temp = dl.attribute(data.to(device), baseline.to(device), target=preds.to(device), return_convergence_delta=True)[0].cpu().detach()
        #temp = dl.attribute(data.to(device), target=preds.to(device)).cpu().detach()
           
        if temp_atts == None: temp_atts = temp
        else:
            temp_atts = torch.cat((temp_atts, temp), 0)
        #if counter % 10 == 0: print(str(counter))
        counter +=1 
        if batches != None:
            if counter == batches: break
    attributions = temp_atts
    
    mean_attributions = np.zeros((len(prediction_names), X.shape[1]))
    for i in range(len(prediction_names)):
        mean_attributions[i] = torch.mean(attributions[classes[i][classes[i] < attributions.shape[0]],:], 0)
    
    mean_attributions = mean_attributions.T
    att_df = pd.DataFrame(mean_attributions)
    att_df.index = genes
    att_df.columns = prediction_names

    return att_df