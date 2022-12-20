from .sc_sharp import scSHARP
import multiprocessing
import itertools
import os
from pkg_resources import resource_filename


class GridSearch:
    """Class for running a grid search on scSHARP consensus cell prediction model"""
    def __init__(self, scSHARP):
        self.sharp = scSHARP
    
    
    def model_grid_search(self, n_workers, random_inits,
        configs='all',
        batch_size=[20, 35, 50, 65, 80, 95],
        neighbors=[10, 50, 100, 250],
        dropouts=[0.0]):
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
        # jobs = []
        self.sharp.random_inits = random_inits
        if configs == 'all':
            configs = os.listdir(resource_filename(__name__, 'configs'))
        
        chunks = self.__get_config_chunks(n_workers, configs, batch_size, neighbors, dropouts)
        pool = multiprocessing.Pool()
        results = pool.map(single_process_search, chunks)
        pool.close()
        pool.join()
        print(results)
        # Now combine the results
        sorted_results = reversed(sorted(results, key=lambda x: x[0]))
        # print(next(sorted_results))
        return sorted_results

    def __get_config_chunks(
        self,  
        chunks,
        configs=os.listdir(resource_filename(__name__, 'configs')),
        batch_size=[20, 35, 50, 65, 80, 95],
        neighbors=[10, 50, 100, 250],
        dropouts=[0.0]):
        """Generates all configs and separates them into chunks for parallel grid search"""

        perms =  list(itertools.product(configs, batch_size, neighbors, dropouts, [self.sharp]))
        return [perms[i::chunks] for i in range(chunks)]



def single_process_search(chunk):
    """Runs training and evaluation for a single hyperparameter configuration
    
    Must remain outside of GridSearch class because multiprocess pool.map does not allow for pickling of class functions.
    """
    
    if len(chunk) == 0:
        return None, None
    
    sharp_ref = chunk[0][4]
    sharp = scSHARP(sharp_ref.data_path, sharp_ref.preds_path, sharp_ref.tools, sharp_ref.marker_path)

    best_acc = 0
    best_config = None
    for config, batch_size, neighbors, dropout, _ in chunk:
        acc = sharp.model_eval(config, batch_size, neighbors, dropout, sharp_ref.random_inits)
        if acc > best_acc:
            best_acc = acc
            best_config = {
            'config':config,
            'batch_size':batch_size,
            'neighbors':neighbors,
            'dropout':dropout}

    # alternatively, we could return all results of the search      
    return best_acc, best_config


