from .sc_sharp import scSHARP, ComponentPredictionsException
import multiprocessing
import itertools
import os
from pkg_resources import resource_filename


class GridSearch:
    """Class for running a grid search on scSHARP consensus cell prediction model"""
    def __init__(self, sharp):
        self.sharp = sharp
        if self.sharp.preds_path == None:
            raise ComponentPredictionsException('Grid search requires running component tools. Execute sharp.run_tools() or provide existing predictions file during initialization')
    
    
    def model_grid_search(self, n_workers, random_inits,
        configs='all',
        batch_size=[20, 35, 50, 65, 80, 95],
        neighbors=[10, 50, 100, 250],
        dropouts=[0.0],
        training_epochs=150):
        """Runs grid search on model to find optimal hyperparameters for dataset
        
        Parameters
        ----------
        n_workers: int
            number of cpu cores available
        random_inits: int
            Number of random initializations to test per model configuration

        Returns
        -------
        sorted_results: list
            best model configutations sorte by evaluation accuracy

        """
        self.sharp.random_inits = random_inits
        self.training_epochs = training_epochs
        if configs == 'all':
            configs = os.listdir(resource_filename(__name__, 'configs'))
        
        chunks = self.__get_config_chunks(n_workers, configs, batch_size, neighbors, dropouts)
        pool = multiprocessing.Pool()
        results = pool.map(single_process_search, chunks)
        pool.close()
        pool.join()
        print(results)
        results = [r for r in results if r != (None, None)]
        # Now combine the results
        sorted_results = reversed(sorted(results, key=lambda x: x[0]))
        return list(sorted_results)

    def __get_config_chunks(
        self,  
        chunks,
        configs=os.listdir(resource_filename(__name__, 'configs')),
        batch_size=[20, 35, 50, 65, 80, 95],
        neighbors=[10, 50, 100, 250],
        dropouts=[0.0]):
        """Generates all configs and separates them into chunks for parallel grid search"""

        perms =  list(itertools.product(configs, batch_size, neighbors, dropouts, [self.training_epochs], [self.sharp]))
        return [perms[i::chunks] for i in range(chunks)]



def single_process_search(chunk):
    """Runs training and evaluation for a single hyperparameter configuration
    
    Must remain outside of GridSearch class because multiprocess pool.map does not allow for pickling of class functions.
    """
    print(chunk)
    if len(chunk) == 0:
        return None, None
    
    sharp_ref = chunk[0][5]
    sharp = scSHARP(sharp_ref.data_path, sharp_ref.tools, sharp_ref.marker_path, preds_path=sharp_ref.preds_path)

    best_acc = 0
    best_config = None
    for config, batch_size, neighbors, dropout, training_epochs, _ in chunk:
        print(f'hyperparameter search on: config: {config}, batch size: {batch_size}, neighbors: {neighbors}')
        total_acc, val_acc, test_acc  = sharp.model_eval(config, batch_size, neighbors, dropout, sharp_ref.random_inits, training_epochs=training_epochs)
        if val_acc > best_acc:
            best_acc = val_acc
            best_config = {
            'config':config,
            'batch_size':batch_size,
            'neighbors':neighbors,
            'dropout':dropout}

    # alternatively, we could return all results of the search      
    return best_acc, best_config


