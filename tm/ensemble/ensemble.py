from __future__ import annotations

from tm.containers import Dataset
from abc import ABC, abstractmethod
import numpy as np

from tm.workflows import cvbt_path

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from tm.model import ModelSet
    
class EnsembleModel(ABC):

    @abstractmethod
    def estimate(self, dataset:Dataset, model_set:ModelSet):
        """Subclasses must implement this method"""
        # it should estimate a number to attribute to each key in dataset accessible with .get method
        # cvbt_path()
        pass

    @abstractmethod
    def get(self, key:str) -> float:
        """Subclasses must implement this method"""
        # returns the portfolio weight for a key
        pass

    def view(self, **kwargs):
        print("EnsembleModel")
        for k, v in self.pws.items():
            print(f'Portfolio Weight for {k} = {v}')



class IdleEnsembleModel(EnsembleModel):
    
    def __init__(self, v = 1, normalize:bool = True):
        self.v = 1
        self.normalize = normalize
        self.pws = {}

    def estimate(self, dataset:Dataset, model_set:ModelSet):
        """Subclasses must implement this method"""
        # it should estimate a number to attribute to each key in dataset accessible with .get method
        # cvbt_path()
        s = 0
        if self.normalize:
            s = self.v*len(dataset)
        if s == 0: s = 1        
        for k, _ in dataset.items():
            self.pws.update({k: self.v / s})

    def get(self, k:str) -> float:
        return self.pws.get(k, 1)        

class InvVolEnsembleModel(EnsembleModel):
    
    def __init__(self):
        self.pws = {}

    def estimate(self, dataset:Dataset, model_set:ModelSet):
        """Subclasses must implement this method"""
        # it should estimate a number to attribute to each key in dataset accessible with .get method
        # cvbt_path()
        
        keys = []
        vols = []
        for k, data in dataset.items():
            if data.n > 5:
                keys.append(k)
                vols.append(np.mean(np.std(data.y, axis = 0)))
            else:
                keys.append(k)
                vols.append(1e10)

        vols = 1 / np.array(vols)
        vols /= np.sum(vols)
        self.pws = dict(zip(keys, vols))

    def get(self, k:str) -> float:
        return self.pws.get(k, 1)        



class InvVolStratFilterEnsembleModel(EnsembleModel):
    def __init__(self, strat_filter_statistic = 'sharpe', k_folds:int = 4, seq_path:bool = False, burn_fraction:float = 0.1, min_burn_points:int = 3):
        self.strat_filter_statistic = strat_filter_statistic
        self.k_folds = k_folds
        self.seq_path = seq_path
        self.burn_fraction = burn_fraction
        self.min_burn_points = min_burn_points
        self.pws = {}

    def _compute_filter_stat(self, s):        
        if self.strat_filter_statistic == 'sharpe':
            scale = np.std(s)
            if scale != 0:
                return np.mean(s) / scale
            else:
                return 0            
        elif self.strat_filter_statistic == 'mean':
            return np.mean(s)
        elif self.strat_filter_statistic == 'gr':
            return np.mean(s) - 0.5*np.mean(np.power(s, 2))
        else:
            raise Exception('Unknown strat_filter_statistic')

    def estimate(self, dataset:Dataset, model_set:ModelSet):
        """Subclasses must implement this method"""
        # it should estimate a number to attribute to each key in dataset accessible with .get method
        # cvbt_path()

        # maybe copies not necessary
        dataset_ = cvbt_path(
                    dataset = dataset.copy(), 
                    modelset = model_set.copy(),
                    k_folds = self.k_folds, 
                    seq_path = self.seq_path, 
                    start_fold = 0, 
                    burn_fraction = self.burn_fraction, 
                    min_burn_points = self.min_burn_points
                    )
        keys = []
        f = []
        w = []
        for k, data in dataset_.items():
            if data.n > 5:
                keys.append(k)
                f.append(self._compute_filter_stat(data.s))
                w.append(1 / np.mean(np.std(data.y, axis = 0)))
            else:
                keys.append(k)
                f.append(-1)
                w.append(1 / 1e10)

        f = np.array(f)
        w = np.array(w)
        w[f<=0] = 0

        d = np.sum(w)        
        if d == 0: w = np.ones_like(w)       
        w /= np.sum(w)
        self.pws = dict(zip(keys, w))

    def get(self, k:str) -> float:
        return self.pws.get(k, 1)      



class EqWStratFilterEnsembleModel(EnsembleModel):
    def __init__(self, strat_filter_statistic = 'sharpe', k_folds:int = 4, seq_path:bool = False, burn_fraction:float = 0.1, min_burn_points:int = 3):
        self.strat_filter_statistic = strat_filter_statistic
        self.k_folds = k_folds
        self.seq_path = seq_path
        self.burn_fraction = burn_fraction
        self.min_burn_points = min_burn_points
        self.pws = {}

    def _compute_filter_stat(self, s):        
        if self.strat_filter_statistic == 'sharpe':
            scale = np.std(s)
            if scale != 0:
                return np.mean(s) / scale
            else:
                return 0            
        elif self.strat_filter_statistic == 'mean':
            return np.mean(s)
        elif self.strat_filter_statistic == 'gr':
            return np.mean(s) - 0.5*np.mean(np.power(s, 2))
        else:
            raise Exception('Unknown strat_filter_statistic')

    def estimate(self, dataset:Dataset, model_set:ModelSet):
        """Subclasses must implement this method"""
        # it should estimate a number to attribute to each key in dataset accessible with .get method
        # cvbt_path()

        # maybe copies not necessary
        dataset_ = cvbt_path(
                    dataset = dataset.copy(), 
                    modelset = model_set.copy(),
                    k_folds = self.k_folds, 
                    seq_path = self.seq_path, 
                    start_fold = 0, 
                    burn_fraction = self.burn_fraction, 
                    min_burn_points = self.min_burn_points
                    )
        keys = []
        f = []
        w = []
        for k, data in dataset_.items():
            if data.n > 5:
                keys.append(k)
                f.append(self._compute_filter_stat(data.s))
                w.append(1.)
            else:
                keys.append(k)
                f.append(-1)
                w.append(1.)

        f = np.array(f)
        w = np.array(w)
        w[f<=0] = 0

        d = np.sum(w)        
        if d == 0: w = np.ones_like(w)       
        w /= np.sum(w)
        self.pws = dict(zip(keys, w))

    def get(self, k:str) -> float:
        return self.pws.get(k, 1)      




class StratStatEnsembleModel(EnsembleModel):
    
    def __init__(self, statistic = 'invvol', k_folds:int = 4, seq_path:bool = False, burn_fraction:float = 0.1, min_burn_points:int = 3):
        self.statistic = statistic        
        self.k_folds = k_folds
        self.seq_path = seq_path
        self.burn_fraction = burn_fraction
        self.min_burn_points = min_burn_points
        self.pws = {}


    def _compute_stat(self, s):
        if self.statistic == 'invvol':
            scale = np.std(s)
            if scale != 0:
                return np.sign(np.mean(s)) / scale
            else:
                return 0
        
        elif self.statistic == 'sharpe':
            scale = np.std(s)
            if scale != 0:
                return np.mean(s) / scale
            else:
                return 0            
        
        elif self.statistic == 'mean':
            return np.mean(s)

        elif self.statistic == 'kelly':
            v = np.var(s)
            if v != 0:
                return np.mean(s) / v
            else:
                return 0     


        elif self.statistic == 'gr':
            return np.mean(s) - 0.5*np.mean(np.power(s, 2))


    def estimate(self, dataset:Dataset, model_set:ModelSet):
        """Subclasses must implement this method"""
        # it should estimate a number to attribute to each key in dataset accessible with .get method
        # cvbt_path()

        # maybe copies not necessary
        dataset_ = cvbt_path(
                    dataset = dataset.copy(), 
                    modelset = model_set.copy(),
                    k_folds = self.k_folds, 
                    seq_path = self.seq_path, 
                    start_fold = 0, 
                    burn_fraction = self.burn_fraction, 
                    min_burn_points = self.min_burn_points
                    )
        keys = []
        w = []
        for k, data in dataset_.items():
            keys.append(k)
            w.append(self._compute_stat(data.s))
        w = np.array(w)
        w[w <= 0] = 0
        d = np.sum(w)        
        if d == 0: w = np.ones_like(w)       
        w /= np.sum(w)
        self.pws = dict(zip(keys, w))

    def get(self, k:str) -> float:
        return self.pws.get(k, 1)        




